"""
Adapted from official swav implementation: https://github.com/facebookresearch/swav
"""
import os
import sys
import ast
from argparse import ArgumentParser, Action
from time import sleep
import math
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import distributed as dist
from torch import nn
from torch.utils import model_zoo
import torch.nn.functional as F
from torchmetrics import Accuracy

from pl_bolts.optimizers.lars import LARS
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)
from pytorch_lightning.utilities.cloud_io import load as load_checkpoint

try:
    from supervised_swav_resnet import resnet18, resnet50, ResNet, BasicBlock
except:
    from .supervised_swav_resnet import resnet18, resnet50, ResNet, BasicBlock

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',     # from Mar 2021
    'resnet50_old': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', # from Jan 2017, used in e.g. FixBi
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def linear_warmup_decay(warmup_steps, total_steps, decay_type='cosine', alpha=10, beta=0.75):
    assert decay_type in ['constant', 'cosine', 'linear', 'inv_prop']

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if decay_type == 'constant':
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if decay_type == 'cosine':
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif decay_type == 'inv_prop':
            # "inversly proportional" decay
            return (1 + alpha*progress)**-beta
        else:
            # linear decay
            return 1.0 - progress

    return fn

class SupervisedSwAV(pl.LightningModule):

    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
        num_classes: int,
        supervised_hidden_mlp: int = 2048,
        supervised_weight: float = 1.,
        share_hidden_mlp: bool = False,
        supervised_head_after_proj_head: bool = False,
        imagenet_pretrained: bool = False,
        old_pretrained_weights: bool = False,
        pretrained_checkpoint: str = None,
        backbone_lr_scaling: float = 1.0,
        val_names: list = None,
        num_nodes: int = 1,
        arch: str = 'resnet50',
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        nmb_prototypes: int = 3000,
        freeze_prototypes_epochs: int = 1,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        queue_length: int = 0,  # must be divisible by total batch-size
        epoch_queue_starts: int = 15,
        crops_for_assign: list = [0, 1],
        nmb_crops: list = [2, 6],
        first_conv: bool = True,
        maxpool1: bool = True,
        norm_layer: str = 'batch_norm',
        num_groups: int = 16,
        optimizer: str = 'adam',
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        lr_schedule_type: str = 'cosine',
        weight_decay: float = 1e-6,
        epsilon: float = 0.05,
        prototype_entropy_regularization_weight: float = 1.,
        prototype_entropy_regularization_type: str = 'separate_optimizer',
        supervised_contrastive: bool = False,
        **kwargs
    ):
        """
        Args:
            gpus: number of gpus per node used in training, passed to SwAV module
                to manage the queue and select distributed sinkhorn
            num_nodes: number of nodes to train on
            num_samples: number of image samples used for training
            batch_size: batch size per GPU in ddp
            dataset: dataset being used for train/val
            num_classes: number of classes in the dataset
            supervised_hidden_mlp: hidden layer of non-linear supervised learning head,
                set to 0 to use a linear projection head
            supervised_weight: weighting factor to multiply supervised loss with
                before adding it to the total loss
            share_hidden_mlp: Whether supervised head and swav's projection head should share their hidden layer. If this is set to True,
                `supervised_hidden_mlp` will have no effect, and `hidden_mlp` will define the size of the hidden layer used by both heads.
            imagenet_pretrained: Whether to initialize the weights of the backbone to values pretrained on imagenet
            old_pretrained_weights: If `imagenet_pretrained` is true: whether to use older pretrained weights from Jan 2017, or newer ones from Mar 2021.
                Otherwise: no effect.
            pretrained_checkpoint: checkpoint to load pretrained weights from. If none is provided, one from pytorch's model_zoo will be used
            val_names: in case of multiple validation dataloaders, this list should
                contain names identifying them (in the same order)
            arch: encoder architecture used for pre-training
            hidden_mlp: hidden layer of non-linear projection head, set to 0
                to use a linear projection head
            feat_dim: output dim of the projection head
            warmup_epochs: apply linear warmup for this many epochs
            max_epochs: epoch count for pre-training
            nmb_prototypes: count of prototype vectors
            freeze_prototypes_epochs: epoch till which gradients of prototype layer
                are frozen
            temperature: loss temperature
            sinkhorn_iterations: iterations for sinkhorn normalization
            queue_length: set queue when batch size is small,
                must be divisible by total batch-size (i.e. total_gpus * batch_size),
                set to 0 to remove the queue
            epoch_queue_starts: start uing the queue after this epoch
            crops_for_assign: list of crop ids for computing assignment
            nmb_crops: number of global and local crops, ex: [2, 6]
            first_conv: keep first conv same as the original resnet architecture,
                if set to false it is replace by a kernel 3, stride 1 conv (cifar-10)
            maxpool1: keep first maxpool layer same as the original resnet architecture,
                if set to false, first maxpool is turned off (cifar10, maybe stl10)
            norm_layer: the type of normalization layer to use in the network (batch_norm or group_norm)
            num_groups: in case group_norm is chosen as norm_layer, this sets the number of groups
            optimizer: optimizer to use
            start_lr: starting lr for linear warmup
            learning_rate: learning rate
            final_lr: final learning rate for cosine weight decay
            lr_schedule_type: type of learning rate schedule to use (constant, linear, cosine or inv_prop)
            weight_decay: weight decay for optimizer
            epsilon: epsilon val for swav assignments
            supervised_head_after_proj_head: Whether to change the architecture such that
                the supervised head attaches after swav's projection head and the l2norm
            backbone_lr_scaling: factor to scale the backbone learning rate with
            prototype_entropy_regularization_weight: weight of the prototype entropy regularization
                (0 for no regularization)
            prototype_entropy_regularization_type: type prototype entropy regularization
                (same_optimizer or separate_optimizer)
            supervised_contrastive: whether to use the supervised contrastive objective,
                using all samples of the same class as positives for the swav loss 
        """
        super().__init__()
        self.save_hyperparameters()

        if nmb_prototypes == 0 and prototype_entropy_regularization_weight != 0:
            raise ValueError('Prototype entropy regularization is not possible without prototypes')

        if not supervised_head_after_proj_head and prototype_entropy_regularization_weight != 0:
            raise ValueError('Prototype entropy regularization requires supervised_head_after_projection_head-architecture')

        if sum(nmb_crops) <= 1:
            raise ValueError('Need at least 2 crops to perform swapped prediction')

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.nmb_prototypes = nmb_prototypes
        self.freeze_prototypes_epochs = freeze_prototypes_epochs
        self.sinkhorn_iterations = sinkhorn_iterations

        self.queue_length = queue_length
        self.epoch_queue_starts = epoch_queue_starts
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops

        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.norm_layer = norm_layer
        self.num_groups = num_groups

        self.optim = optimizer
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.lr_schedule_type = lr_schedule_type
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.supervised_hidden_mlp = supervised_hidden_mlp
        self.supervised_weight = supervised_weight
        self.num_classes = num_classes
        self.share_hidden_mlp = share_hidden_mlp
        self.supervised_head_after_proj_head = supervised_head_after_proj_head

        self.val_names = val_names
        self.imagenet_pretrained = imagenet_pretrained
        self.pretrained_checkpoint = pretrained_checkpoint
        self.old_pretrained_weights = old_pretrained_weights
        self.backbone_lr_scaling = backbone_lr_scaling

        self.prototype_entropy_regularization_weight = prototype_entropy_regularization_weight
        self.prototype_entropy_regularization_type = prototype_entropy_regularization_type

        self.supervised_contrastive = supervised_contrastive

        if self.gpus * self.num_nodes > 1:
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn

        self.model = self.init_model()

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        self.queue = None
        self.use_the_queue = False
        self.softmax = nn.Softmax(dim=1)

        # metrics
        self.train_acc = Accuracy()
        if val_names:
            for name in val_names:
                setattr(self, f'{name}_val_acc', Accuracy(compute_on_step=False))
        else:
            self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)

    def init_model(self):
        if self.arch == 'resnet18':
            backbone = resnet18
            initial_filters = 64
        elif self.arch == 'resnet18_w32':
            backbone = resnet18
            initial_filters = 32
        elif self.arch == 'resnet18_w16':
            backbone = resnet18
            initial_filters = 16
        elif self.arch == 'resnet50':
            backbone = resnet50
            initial_filters = 64
        elif self.arch == 'resnet50_w32':
            backbone = resnet50
            initial_filters = 32
        elif self.arch == 'resnet50_w16':
            backbone = resnet50
            initial_filters = 16
        elif self.arch == 'resnet32_w16':
            # the ResNet as used in the resnet paper (https://arxiv.org/pdf/1512.03385.pdf) for cifar10 classification, with n=5
            # note that they did not use a dense layer in the classifiation head, corresponding to self.supervised_hidden_mlp = 0
            backbone = lambda **kwargs: ResNet(BasicBlock, [5, 5, 5], **kwargs)
            initial_filters = 16
        elif self.arch == 'resnet26_w32':
            # as used in the MT3 paper
            backbone = lambda **kwargs: ResNet(BasicBlock, [4, 4, 4], **kwargs)
            initial_filters = 32
        else:
            raise ValueError('Invalid architecture name')

        if self.norm_layer == 'batch_norm':
            norm_layer = nn.BatchNorm2d
        elif self.norm_layer == 'group_norm':
            norm_layer = lambda num_channels: nn.GroupNorm(self.num_groups, num_channels)
        else:
            raise ValueError('Invalid norm layer name')

        model = backbone(
            num_classes=self.num_classes,
            supervised_hidden_mlp=self.supervised_hidden_mlp,
            share_hidden_mlp=self.share_hidden_mlp,
            width_per_group=initial_filters,
            norm_layer=norm_layer,
            normalize=True,
            hidden_mlp=self.hidden_mlp,
            output_dim=self.feat_dim,
            nmb_prototypes=self.nmb_prototypes,
            first_conv=self.first_conv,
            maxpool1=self.maxpool1,
            supervised_head_after_proj_head=self.supervised_head_after_proj_head,
        )
        if self.imagenet_pretrained:
            if self.arch != 'resnet50' or not self.first_conv or not self.maxpool1:
                raise ValueError('Pretrained weights are only available for the resnet50 with first_conv and maxpool1 = True')
            if self.pretrained_checkpoint is None:
                missing_keys, unexpected_keys = model.load_state_dict(model_zoo.load_url(model_urls['resnet50' + ('_old' if self.old_pretrained_weights else '')]), strict=False)
            else:
                checkpoint_state_dict = load_checkpoint(self.pretrained_checkpoint)['state_dict']
                # remove the prefix "model." from keys, if present
                checkpoint_state_dict = OrderedDict([(k[6:], v) if k.startswith('model.') else (k, v) for k, v in checkpoint_state_dict.items()])
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state_dict, strict=False)
            print('Missing keys while loading state dict:', missing_keys)
            print('Unexpected keys while loading state dict:', unexpected_keys)
        return model

    def forward(self, x):
        # pass single batch through resnet backbone and supervised head
        return self.model(x)

    def forward_backbone(self, x):
        # pass single batch from the resnet backbone
        return self.model.forward_backbone(x)

    def forward_embeddings(self, x):
        # pass single batch through resnet backbone and projection head (including normalization)
        result = self.model.forward_swav(x)
        return result[0] if isinstance(result, tuple) else result

    def get_prototypes(self):
        return self.model.prototypes.weight if self.model.prototypes is not None else None

    def on_train_epoch_start(self):
        if self.queue_length > 0:
            if self.trainer.current_epoch >= self.epoch_queue_starts and self.queue is None:
                self.queue = torch.zeros(
                    len(self.crops_for_assign),
                    self.queue_length // self.gpus,  # change to nodes * gpus once multi-node
                    self.feat_dim,
                )

                if self.gpus > 0:
                    self.queue = self.queue.cuda()

        self.use_the_queue = False

    def on_after_backward(self):
        if self.current_epoch < self.freeze_prototypes_epochs:
            for name, p in self.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

    def shared_step(self, batch, acc_metric): #, conf_mat):
        if self.dataset == 'stl10':
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        inputs, labels = batch

        # last element of inputs: batch used for supervised training
        # rest of the inputs: multicrop-batches for SwAV
        # if inputs contains only one element, only supervised learning is performed
        # if inputs contains exactly sum(self.nmb_crops) elements, only swav is performed
        # otherwise, an error is raised

        if len(inputs) not in (1, sum(self.nmb_crops), sum(self.nmb_crops) + 1):
            raise ValueError(
                f'Got inputs of invalid length {len(inputs)}, when either length 1 for supervised only, '
                f'length {sum(self.nmb_crops)} for swav only, or length {sum(self.nmb_crops) + 1} for both jointly was expected.'
            )

        swav_loss = 0

        if len(inputs) > 1:
            ## SwAV

            # 1. normalize the prototypes
            with torch.no_grad():
                w = self.model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.model.prototypes.weight.copy_(w)

            # 2. multi-res forward passes
            if len(inputs) == sum(self.nmb_crops):
                # swav only
                embedding, output = self.model.forward_swav(inputs)
            else:
                # supervised + swav
                logits, (embedding, output) = self.model(inputs)
            embedding = embedding.detach()
            bs = inputs[0].size(0)

            # 3. swav loss computation
            for i, crop_id in enumerate(self.crops_for_assign):
                with torch.no_grad():
                    out = output[bs * crop_id:bs * (crop_id + 1)]

                    # 4. time to use the queue
                    if self.queue is not None:
                        if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                            self.use_the_queue = True
                            out = torch.cat((torch.mm(self.queue[i], self.model.prototypes.weight.t()), out))
                        # fill the queue
                        self.queue[i, bs:] = self.queue[i, :-bs].clone()
                        self.queue[i, :bs] = embedding[crop_id * bs:(crop_id + 1) * bs]

                    # 5. get assignments
                    q = torch.exp(out / self.epsilon).t()
                    q = self.get_assignments(q, self.sinkhorn_iterations)[-bs:]

                # output = [crop_0_of_sample_0, crop_0_of_sample_1, ..., crop_1_of_sample_0, crop_1_of_sample_1, ..., ...]
                #           <---------- crop 0 of the batch ---------->, <---------- crop 1 of the batch ---------->, ...

                # cluster assignment prediction
                if self.supervised_contrastive:
                    p = self.softmax(output / self.temperature) # generate the p-term in the swav loss for all crops of all images in the batch

                    # matrix of size (bs, sum(self.nmb_crops) * bs), which contains the potential losses for the prediction of all the cosine_similarites by all the assignments in the batch (for this crop)
                    potential_sublosses = -torch.mm(q, torch.log(p).t())

                    # the value at position i,j of this matrix defines, whether the i'th soft assignment (q) should be matched with the j'th cosine_similarity
                    # i is in [0, bs), while j is in [0, sum(self.nmb_crops) * bs)
                    # because assignments are only made for one crop per loop iteration, and cosine_similarities (`output`/`p`) are available for all crops at the same time
                    classes_matching_index_not_matching = torch.cat([torch.stack([labels == label for label in labels]) for _ in range(sum(self.nmb_crops))], dim=1)

                    # don't match to the exact same image itself (note that this only excludes this exact crop of the image, not the other crops of multi-crop)
                    for i in range(bs):
                        classes_matching_index_not_matching[i, crop_id*bs + i] = False

                    swav_loss += torch.mean(potential_sublosses[classes_matching_index_not_matching])

                else:
                    subloss = 0
                    for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                        p = self.softmax(output[bs * v:bs * (v + 1)] / self.temperature)
                        subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
                    swav_loss += subloss / (np.sum(self.nmb_crops) - 1)

            swav_loss /= len(self.crops_for_assign)
        else:
            # supervised only
            logits = self.model(inputs)

        ce_loss = 0
        if len(inputs) != sum(self.nmb_crops):
            ## Supervised
            ce_loss = F.cross_entropy(logits, labels)
            acc_metric(logits, labels)

        joint_loss = swav_loss + self.supervised_weight * ce_loss

        return ce_loss, swav_loss, joint_loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if optimizer_idx == 0:
            ce_loss, swav_loss, joint_loss = self.shared_step(batch, self.train_acc) #, self.train_conf_mat)

            self.log('train/ce_loss', ce_loss, on_step=True, on_epoch=False)
            self.log('train/swav_loss', swav_loss, on_step=True, on_epoch=False)
            self.log('train/joint_loss', joint_loss, on_step=True, on_epoch=False)

            self.log('train/acc', self.train_acc, prog_bar=True)

            if self.prototype_entropy_regularization_type == 'same_optimizer' and self.prototype_entropy_regularization_weight != 0:
                joint_loss = joint_loss + self.prototype_entropy_regularization_weight * self.get_regularization_term()

            return joint_loss
        elif optimizer_idx == 1:
            return self.get_regularization_term()

    def get_regularization_term(self):
        prototype_classifications = self.softmax(
            self.model.supervised_head(self.model.prototypes.weight)
        )
        # average entropy of each individual prototype classification (marginal entropy)
        # -> minimize to have prototypes that closely resemble actual classes
        proto_entropy = torch.mean(-torch.sum(prototype_classifications * torch.log(prototype_classifications), dim=1))
        self.log('train/prototype_classification_entropy', proto_entropy, on_step=True, on_epoch=False)

        # entropy of all classifcations averaged -> maximize to have each class represented an equal number of times
        classification_mean = torch.mean(prototype_classifications, dim=0)
        mean_classification_entropy = -torch.sum(classification_mean * torch.log(classification_mean))
        self.log('train/proto_mean_classification_entropy', mean_classification_entropy, on_step=True, on_epoch=False)

        return proto_entropy - mean_classification_entropy

    def validation_step(self, batch, batch_idx, dataloader_id=None):
        if dataloader_id is not None:
            dataloader_name = self.val_names[dataloader_id]

        ce_loss, swav_loss, joint_loss = self.shared_step(
            batch,
            self.val_acc if dataloader_id is None else getattr(self, f'{dataloader_name}_val_acc')
        )

        slash_dataloader_name = f'/{dataloader_name}' if dataloader_id is not None else ''
        self.log(f'val{slash_dataloader_name}/ce_loss', ce_loss, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f'val{slash_dataloader_name}/swav_loss', swav_loss, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f'val{slash_dataloader_name}/joint_loss', joint_loss, on_step=False, on_epoch=True, add_dataloader_idx=False)

        self.log(
            f'val{slash_dataloader_name}/acc',
            self.val_acc if dataloader_id is None else getattr(self, f'{dataloader_name}_val_acc'),
            add_dataloader_idx=False
        )

        return joint_loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        assert not isinstance(inputs, list), "Test data should not be multi-cropped"

        logits = self.model(inputs)
        ce_loss = F.cross_entropy(logits, labels)
        self.test_acc(logits, labels)

        self.log('test/ce_loss', ce_loss, on_step=False, on_epoch=True)
        self.log('test/acc', self.test_acc)

        return ce_loss

    def scale_backbone_lr(self, named_params, learning_rate, scale_factor=0.1):
        backbone_params = []
        head_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in ('shared_head', 'supervised_head', 'projection_head', 'prototypes')):
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [{'params': head_params, 'lr': learning_rate}, {'params': backbone_params, 'lr': learning_rate*scale_factor}]

    def configure_optimizers(self):
        if self.backbone_lr_scaling != 1:
            params = self.scale_backbone_lr(self.named_parameters(), learning_rate=self.learning_rate, scale_factor=self.backbone_lr_scaling)
        else:
            params = self.parameters()

        if self.optim == 'lars':
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, decay_type=self.lr_schedule_type),
            ),
            "interval": "step",
            "frequency": 1,
        }

        if self.prototype_entropy_regularization_type == 'separate_optimizer' and self.prototype_entropy_regularization_weight != 0:
            prototype_parameters = [param for name, param in self.named_parameters() if 'prototypes' in name]
            regularization_lr = self.learning_rate * self.prototype_entropy_regularization_weight

            if self.optim == 'lars':
                prototype_entropy_optimizer = LARS(
                    prototype_parameters,
                    lr=regularization_lr,
                    momentum=0.9,
                    weight_decay=self.weight_decay,
                    trust_coefficient=0.001,
                )
            elif self.optim == 'adam':
                prototype_entropy_optimizer = torch.optim.Adam(prototype_parameters, lr=regularization_lr, weight_decay=self.weight_decay)
            elif self.optim == 'sgd':
                prototype_entropy_optimizer = torch.optim.SGD(prototype_parameters, lr=regularization_lr, momentum=0.9, weight_decay=self.weight_decay)

            prototype_entropy_scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    prototype_entropy_optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, decay_type=self.lr_schedule_type),
                ),
                "interval": "step",
                "frequency": 1,
            }

            return [optimizer, prototype_entropy_optimizer], [scheduler, prototype_entropy_scheduler]

        return [optimizer], [scheduler]

    def sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)

                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @staticmethod
    def add_model_specific_args(parent_parser):
        def true_false(arg):
            if arg == 'True':
                return True
            elif arg == 'False':
                return False
            else:
                raise ValueError()

        def list_from_string(dtype):
            def f(arg):
                val_list = ast.literal_eval(arg)
                if type(val_list) is not list or len(val_list) == 0:
                    raise ValueError(f'{arg} is an invalid representation for a list')
                # check whether all elements can be interpreted as an object of type dtype
                for x in val_list:
                    dtype(x)
                return val_list

            f.__name__ = f'{dtype.__name__}_list'
            return f

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
        parser.add_argument("--imagenet_pretrained", default=False, type=true_false,
            help="Whether to initialize convnet with pretrained imagenet weights. Only available for resnet50"
        )
        parser.add_argument('--pretrained_checkpoint', default=None, type=str,
            help="checkpoint to load pretrained weights from. If none is provided, one from pytorch's model_zoo will be used"
        )
        parser.add_argument('--old_pretrained_weights', default=False, type=true_false,
            help='If --imagenet_pretrained is True: whether to use older pretrained weights from Jan 2017, or newer ones from Mar 2021. Otherwise: no effect.'
        )
        # specify flags to store false
        parser.add_argument("--first_conv", action='store_false')
        parser.add_argument("--maxpool1", action='store_false')
        parser.add_argument("--norm_layer", type=str, default='batch_norm', action='store', help='batch_norm, group_norm')
        parser.add_argument("--num_groups", type=int, default=16, action='store',
            help='in case group_norm is chosen as norm_layer, this sets the number of groups'
        )
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--fp32", action='store_true')

        # transform params
        parser.add_argument("--gaussian_blur", type=true_false, action='store', default=False, help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10, cifar100, office31, imagenet")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

        parser.add_argument(
            "--nmb_crops", type=list_from_string(int), default=[2, 4], help="list of number of crops (example: [2, 6])"
        )
        parser.add_argument(
            "--size_crops", type=list_from_string(int), default=[96, 36], help="crops resolutions (example: [224, 96])"
        )
        parser.add_argument(
            "--min_scale_crops",
            type=list_from_string(float),
            default=[0.33, 0.10],
            help="argument in RandomResizedCrop (example: [0.14, 0.05])",
        )
        parser.add_argument(
            "--max_scale_crops",
            type=list_from_string(float),
            default=[1, 0.33],
            help="argument in RandomResizedCrop (example: [1., 0.14])",
        )

        # training params
        parser.add_argument("--fast_dev_run", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars/sgd")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")
        parser.add_argument("--backbone_lr_scaling", type=float, default=1.0, help="factor to scale the backbone learning rate with")
        parser.add_argument('--lr_schedule_type', default='cosine', type=str, choices=['constant', 'linear', 'cosine', 'inv_prop'])

        # swav params
        parser.add_argument(
            "--crops_for_assign",
            type=list_from_string(int),
            default=[0, 1],
            help="list of crops id used for computing assignments",
        )
        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument(
            "--epsilon", default=0.05, type=float, help="regularization parameter for Sinkhorn-Knopp algorithm"
        )
        parser.add_argument(
            "--sinkhorn_iterations", default=3, type=int, help="number of iterations in Sinkhorn-Knopp algorithm"
        )
        parser.add_argument("--nmb_prototypes", default=512, type=int, help="number of prototypes")
        parser.add_argument(
            "--queue_length",
            type=int,
            default=0,
            help="length of the queue (0 for no queue); must be divisible by total batch size"
        )
        parser.add_argument(
            "--epoch_queue_starts", type=int, default=15, help="from this epoch, we start using a queue"
        )
        parser.add_argument(
            "--freeze_prototypes_epochs",
            default=1,
            type=int,
            help="freeze the prototypes during this many epochs from the start"
        )

        parser.add_argument('--supervised_only', type=true_false, action='store', default=False, help='train only using the supervised head')
        parser.add_argument('--supervised_hidden_mlp', type=int, default=2048, help='hidden layer of non-linear supervised learning head')
        parser.add_argument('--supervised_weight', type=float, default=1., help='weighting factor of supervised loss')
        parser.add_argument('--supervised_transforms', type=str, default='default',
            choices=['default', 'larger_crop', 'cifar10_transforms'], help='default, larger_crop, cifar10_transforms'
        )
        parser.add_argument('--supervised_crop_size', type=int, default=-1,
            help=('(square) size to crop the images used for supervised training to. '
                  'Defaults to the larger one of the multi-crop sizes (usually a sensible choice for best alignment between the swav and supervised objectives). '
                  'Needs to be specified for supervised only training. '
                  'Note that "cifar10_transforms" ignores this argument and always crops to 32x32.')
        )

        parser.add_argument('--share_hidden_mlp', type=true_false, default=False,
            help=("Whether supervised head and swav's projection head should share their hidden layer. If this is set to True, "
                  '"--supervised_hidden_mlp" will have no effect, and "--hidden_mlp" will define the size of the hidden layer used by both heads.')
        )
        parser.add_argument('--supervised_head_after_proj_head', type=true_false, default=False,
            help=("Whether to change the architecture such that the supervised head attaches after swav's projection head and the l2norm, "
                  'making the projection head part of the shared parameters. This makes the inputs of the supervised head be in the same space '
                  'as the prototypes, which is necessary to perform prototype entropy regularization. '
                  'If this is set to True, "--share_hidden_mlp" must be set to False.')
        )

        parser.add_argument('--prototype_entropy_regularization_weight', type=float, default=0,
            help=('Weight of the regularization term that minimizes marginal prototype entropy and maximizes mean prototype entropy. '
                  'If a separate optimizer is used, this is simply a factor by which its learning rate is smaller than the first optimizer, '
                  'otherwise its a factor the regularization term gets multiplied with before adding it to the total loss. '
                  'The default of 0 implicates NO regularization. '
                  'If this value is set to >0, "--supervised_head_after_proj_head" is required to be set to True.')
        )
        parser.add_argument('--prototype_entropy_regularization_type', type=str, default='same_optimizer', choices=['separate_optimizer', 'same_optimizer'],
            help='If a separate optimizer is used, this optimizer ONLY optimizes the prototypes, otherwise the supervised head is also affected by the regularization'
        )

        parser.add_argument('--supervised_contrastive', type=true_false, action='store', default=False,
            help='Whether to use labels in order to treat samples with the same label as positives for the swav loss'
        )

        return parser


def cli_main():
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from cifar100_datamodule import CIFAR100DataModule
    from Office31 import Office31DataModule
    from pl_bolts.models.self_supervised.swav.transforms import SwAVEvalDataTransform, SwAVTrainDataTransform
    from pytorch_lightning import loggers as pl_loggers
    from torchvision import transforms as transforms

    parser = ArgumentParser(allow_abbrev=False)

    parser.add_argument('--wandb_project', default='uncategorized', type=str, help='name of the wandb project to save this run in')
    parser.add_argument('--wandb_log_dir', default='wandb_logs', type=str, help='directory to save wandb logs in')
    parser.add_argument('--hyperparameter_checkpoint', type=str,
        help=(
            'A checkpoint can be specified the hyperparameters of which will be extracted and used for this run. '
            'Parameters can still be overwritten using the command line arguments. '
            'Basically, the hyperparameters of this checkpoint act as the default values of the arguments (instead of the hard-coded ones). '
        )
    )
    parser.add_argument('--wandb_run_name', default=None, type=str, help='custom name for this run in wandb')
    parser.add_argument('--hyperparameter_artifact', type=str,
        help='The same as "--hyperparameter_checkpoint", but the checkpoint is specified as a wandb artifact'
    )
    parser.add_argument('--artifact_dir', type=str, action='store', default='wandb_artifacts', help='directory to save wandb artifacts in')
    parser.add_argument('--wandb_entity', type=str, help='name of the wandb entity (account) to load an artifact from')
    parser.add_argument('--logger_prefix', default='supervised_swav', type=str,
        help='prefix to the name of all the logged metrics (empty string for no prefix)',
    )

    # model args
    parser = SupervisedSwAV.add_model_specific_args(parser)
    args = parser.parse_args()

    assert (args.hyperparameter_checkpoint is None) or (args.hyperparameter_artifact is None), \
        'Only hyperparameter_checkpoint or hyperparameter_artifact can be specified'

    if args.hyperparameter_checkpoint is not None:
        assert os.path.isfile(args.hyperparameter_checkpoint), 'The specified checkpoint file does not exist'

    logger = pl_loggers.WandbLogger(
        save_dir=args.wandb_log_dir,
        project=args.wandb_project,
        prefix=args.logger_prefix,
        log_model=True,
    )
    logger.LOGGER_JOIN_CHAR = '/'

    if args.hyperparameter_artifact is not None:
        assert args.wandb_entity is not None, "Need to specify a wandb_entity to load an artifact"
        artifact_dirname = args.hyperparameter_artifact.replace(":", "-")

        if type(logger.experiment) is not pl_loggers.base.DummyExperiment:
            artifact = logger.experiment.use_artifact(f'{args.wandb_entity}/{args.wandb_project}/{args.hyperparameter_artifact}', type='model')
            if not os.path.isdir(os.path.join(args.artifact_dir, artifact_dirname)):
                # artifact has not been downloaded yet -> download artifact and store its path in hyperparameter_checkpoint
                args.hyperparameter_checkpoint = os.path.join(
                    artifact.download(os.path.join(
                        args.artifact_dir,
                        artifact_dirname
                    )),
                    'model.ckpt'
                )
            else:
                # artifact is already downloaded, because it was used before -> just store its path in hyperparameter_checkpoint
                args.hyperparameter_checkpoint = os.path.join(
                    args.artifact_dir,
                    artifact_dirname,
                    'model.ckpt'
                )
        else:
            # we got a DummyExperiment -> we are not rank 0
            # wait until the rank 0 process downloaded the artifact, then use it to obtain the hyperparameters
            while not os.path.isdir(os.path.join(args.artifact_dir, artifact_dirname)):
                pass
            # now the file exists, but it might still be in the process of downloading
            args.hyperparameter_checkpoint = os.path.join(
                args.artifact_dir,
                artifact_dirname,
                'model.ckpt'
            )

    if args.hyperparameter_checkpoint is not None:
        while True:
            try:
                hyperparameters = load_checkpoint(args.hyperparameter_checkpoint)['hyper_parameters']
                break
            except:
                print('Waiting for download...')
                sleep(1)

        # overwrite arguments that were explicitly specified in the command line
        hyperparameters.update({k: v for k, v in args.__dict__.items() if any(f'--{k}' in arg for arg in sys.argv)})
        # * this assumes "_" in argument names, NOT "-"
        # * and it assumes no shortcut names for arguments

        args.__dict__ = hyperparameters

    assert not (args.supervised_only and args.supervised_contrastive), 'No contrastive objective present when training purely supervised'

    logger.experiment.name = '/'.join([
        args.dataset,
        "supervised" if args.supervised_only else "supervised_swav"] +
        ([args.hyperparameter_artifact] if 'hyperparameter_artifact' in args.__dict__ and args.hyperparameter_artifact is not None else ([os.path.basename(args.hyperparameter_checkpoint)] if 'hyperparameter_checkpoint' in args.__dict__ and args.hyperparameter_checkpoint is not None else [])) +
        ([args.wandb_run_name] if args.wandb_run_name else [])
    )

    if args.dataset == 'stl10':
        raise NotImplementedError("stl10 has not yet been implemented for the supervised setting")
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False

        normalization = stl10_normalization()
    elif args.dataset == 'cifar10':
        dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False

        normalization = cifar10_normalization()
    elif args.dataset == 'cifar100':
        dm = CIFAR100DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False

        normalization = cifar10_normalization()
    elif args.dataset.startswith('office31'):
        if len(args.dataset) < 9 or args.dataset[8] != ':':
            raise ValueError('Violated expected format for office31 dataset name: "office31:<domain>", e.g."office31:amazon"')

        train_domain = args.dataset[9:]
        dm = Office31DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, domain=train_domain, val_split=0)
        args.num_samples = dm.num_samples

        normalization = imagenet_normalization()

        # use training sets of the other domains as validation during training on this domain
        val_domains = [domain for domain in ('amazon', 'dslr', 'webcam') if domain != train_domain]
        val_dataloaders = []
        for domain in val_domains:
            val_dm = Office31DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, domain=domain, val_split=0.2)

            val_dm.train_transforms = SwAVEvalDataTransform(
                normalize=normalization,
                size_crops=args.size_crops,
                nmb_crops=args.nmb_crops,
                min_scale_crops=args.min_scale_crops,
                max_scale_crops=args.max_scale_crops,
                gaussian_blur=args.gaussian_blur,
                jitter_strength=args.jitter_strength
            )
            if args.supervised_only:
                val_dm.train_transforms.transform = [val_dm.train_transforms.transform[-1]]
            val_dm.train_transforms.transform[-1] = transforms.Compose([
                transforms.ToTensor(),
                normalization,
            ])
            val_dm.setup(stage='fit')

            val_dataloaders.append(val_dm.train_dataloader())

        dm.val_dataloader = lambda: val_dataloaders
        args.val_names = val_domains

    elif args.dataset == 'imagenet':
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.size_crops = [224, 96]
        args.nmb_crops = [2, 6]
        args.min_scale_crops = [0.14, 0.05]
        args.max_scale_crops = [1., 0.14]
        args.gaussian_blur = True
        args.jitter_strength = 1.

        args.batch_size = 64
        args.num_nodes = 8
        args.gpus = 8  # per-node
        args.max_epochs = 800

        args.optimizer = 'lars'
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3

        args.nmb_prototypes = 3000

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    args.num_classes = dm.num_classes

    if args.supervised_only:
        # if only supervised training is done, don't create swav-specific layers of resnet
        args.nmb_prototypes = 0
        if not args.supervised_head_after_proj_head:
            args.feat_dim = 0

    dm.train_transforms = SwAVTrainDataTransform(
        normalize=normalization,
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength
    )
    if args.supervised_only:
        dm.train_transforms.transform = [dm.train_transforms.transform[-1]]

    if args.supervised_crop_size < 0:
        assert not args.supervised_only, 'For supervised_only, supervised_crop_size needs to be specified'
        args.supervised_crop_size = args.size_crops[0]

    # set the augmentation transforms for supervised training
    if args.supervised_transforms == 'cifar10_transforms':
        args.supervised_crop_size = 32
        # the augmentations used for cifar10 in the resnet paper
        # https://arxiv.org/pdf/1512.03385.pdf
        # or in
        # https://github.com/kentaroy47/pytorch-lightning-tryouts/blob/master/cifar10.py
        dm.train_transforms.transform[-1] = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ])
    elif args.supervised_transforms == 'larger_crop':
        dm.train_transforms.transform[-1] = transforms.Compose([
            transforms.RandomResizedCrop(args.supervised_crop_size, scale=(0.3125, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ])
    else:
        # the default ones as defined in SwAVTrainDataTransform:
        dm.train_transforms.transform[-1] = transforms.Compose([
            transforms.RandomResizedCrop(args.supervised_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ])

    dm.val_transforms = SwAVEvalDataTransform(
        normalize=normalization,
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength
    )
    if args.supervised_only:
        dm.val_transforms.transform = [dm.val_transforms.transform[-1]]

    dm.val_transforms.transform[-1] = transforms.Compose([
        transforms.ToTensor(),
        normalization,
    ])

    # swav model init
    model = SupervisedSwAV(**args.__dict__)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        save_last=not args.dataset.startswith('office31'),
        save_top_k=1,
        monitor='val/joint_loss' if not args.dataset.startswith('office31') else
                ('val/webcam/joint_loss' if args.dataset.endswith('amazon') else 'val/amazon/joint_loss')
    )
    callbacks = [model_checkpoint, lr_monitor]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        distributed_backend='ddp' if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        logger=logger,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
