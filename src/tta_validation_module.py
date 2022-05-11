"""
Adapted from official swav implementation: https://github.com/facebookresearch/swav
"""
import os
import ast
from argparse import ArgumentParser
from time import sleep
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import distributed as dist
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy

from pl_bolts.optimizers.lars import LARS
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
)
from pytorch_lightning.utilities.cloud_io import load as load_checkpoint

import copy

try:
    from supervised_swav_resnet import resnet18, resnet50, ResNet, BasicBlock
except:
    from .supervised_swav_resnet import resnet18, resnet50, ResNet, BasicBlock

class SupervisedSwAVTTAEvaluator(pl.LightningModule):

    def __init__(
        self,
        gpus: int,
        batch_size: int,
        num_classes: int,
        supervised_hidden_mlp: int,
        arch: str,
        hidden_mlp: int,
        feat_dim: int,
        nmb_prototypes: int,
        crops_for_assign: list,
        nmb_crops: list,
        first_conv: bool,
        maxpool1: bool,
        supervised_head_after_proj_head: bool,
        share_hidden_mlp: bool = False,
        adapt_exclusively: Union[str, list] = None,
        loss_type: str = 'swav',
        supervised_weight: float = 1.,
        num_nodes: int = 1,
        num_steps: int = 1,
        freeze_prototypes: bool = True,
        freeze_norm_layers: bool = False,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        norm_layer: str = 'batch_norm',
        num_groups: int = 16,
        optimizer: str = 'adam',
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        weight_decay: float = 1e-6,
        epsilon: float = 0.05,
        q_generator: str = 'softmax_normalized',
        use_whole_batch_as_positives: bool = False,
        **kwargs
    ):
        """
        Args:
            gpus: number of gpus per node
            batch_size: batch size (number of copies being used per adaption step)
            num_classes: number of classes in the dataset
            supervised_hidden_mlp: hidden layer of non-linear supervised learning head,
                set to 0 to use a linear projection head
            arch: encoder architecture used for pre-training
            hidden_mlp: hidden layer of non-linear projection head, set to 0
                to use a linear projection head
            feat_dim: output dim of the projection head
            nmb_prototypes: count of prototype vectors
            crops_for_assign: list of crop ids for computing assignment
            nmb_crops: number of global and local crops, ex: [2, 6]
            first_conv: keep first conv same as the original resnet architecture,
                if set to false it is replace by a kernel 3, stride 1 conv (cifar-10)
            maxpool1: keep first maxpool layer same as the original resnet architecture,
                if set to false, first maxpool is turned off (cifar10, maybe stl10)
            supervised_head_after_proj_head: Whether to change the architecture such that
                the supervised head attaches after swav's projection head and the l2norm
            share_hidden_mlp: Whether supervised head and swav's projection head should share their hidden layer. If this is set to True,
                `supervised_hidden_mlp` will have no effect, and `hidden_mlp` will define the size of the hidden layer used by both heads.
            adapt_exclusively: Which parts of the model to adapt, all other parameters will remain fixed,
                ex: 'layer3' (third block of resnet backbone)
            loss_type: type of loss to use for adaption. 'swav' for swapped prediction, alternatively 'nearest_prototype_distance'
            supervised_weight: weighting factor to multiply supervised loss with before adding it to the total loss,
                for TTA it has no effect other than to make the joint loss comparable to the one logged during training
            num_nodes: number of nodes
            num_steps: the number of gradient steps to take for adaption before making the prediction,
                train_dataloader is expected to provide exactly this many batches of each image
            freeze_prototypes: whether to keep the prototypes frozen during evaluation
                (only makes a difference with num_steps > 1)
            freeze_norm_layers: whether to keep the norm layers frozen during evaluation
            temperature: loss temperature
            sinkhorn_iterations: iterations for sinkhorn normalization
            norm_layer: the type of normalization layer to use in the network (batch_norm or group_norm)
            num_groups: number of groups if group normalization is used
            optimizer: optimizer to use
            start_lr: starting lr for linear warmup
            learning_rate: learning rate
            final_lr: final learning rate for cosine weight decay
            weight_decay: weight decay for optimizer
            epsilon: epsilon val for swav assignments
            q_generator: how to generate the matrix Q from C^T*Z
            use_whole_batch_as_positives: whether to use all views in the batch for the swapped prediction (modification related to SCL)
        """
        super().__init__()
        self.save_hyperparameters()

        if sum(nmb_crops) <= 1:
            raise ValueError('Need at least 2 crops to perform swapped prediction')
        if loss_type not in ('nearest_prototype_distance', 'swav'):
            raise ValueError('Invalid loss type')

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.nmb_prototypes = nmb_prototypes
        self.freeze_prototypes = freeze_prototypes
        self.freeze_norm_layers = freeze_norm_layers
        self.sinkhorn_iterations = sinkhorn_iterations

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

        self.supervised_hidden_mlp = supervised_hidden_mlp
        self.supervised_weight = supervised_weight
        self.num_classes = num_classes
        self.share_hidden_mlp = share_hidden_mlp
        self.supervised_head_after_proj_head=supervised_head_after_proj_head

        self.num_steps = num_steps

        self.loss_type = loss_type
        self.adapt_exclusively = None if adapt_exclusively == 'all' else [adapt_exclusively] if not isinstance(adapt_exclusively, list) else adapt_exclusively

        if self.gpus * self.num_nodes > 1:
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn

        self.softmax = nn.Softmax(dim=1)

        # NOTE: in the following comments, dimension 0 is considered the column dimension and dim 1 is considered the row dimension,
        # so batch size B is the number of columns, and number of prototypes K is the number of rows
        if q_generator == 'sinkhorn':
            # enforces for Q
            # - all values > 0
            # - all columns sum to 1
            # - all rows sum to 1/K ("every prototype is chosen equally likely")
            # NOTE: since all images of the batch are a copy of the same image, choosing every prototype equally doesn't make much sense anymore.
            # Empirical evidence supports this, resulting in sharp drops in accuracy when using this method.
            self.get_q = lambda CTZ: self.get_assignments(torch.exp(CTZ / self.epsilon).t(), self.sinkhorn_iterations)
        elif q_generator == 'softmax_normalized':
            # still enforces for Q
            # - all values > 0
            # - all columns sum to 1
            # doesn't enforce
            # - all rows sum to 1/K
            self.get_q = lambda CTZ: (self.softmax(CTZ/self.epsilon)).detach()
        elif q_generator == 'naively_normalized':
            # adds 1 to make each element positive, then normalizes each column to sum to 1
            self.get_q = self.naively_normalized
        elif q_generator == 'hard_assignment':
            self.get_q = lambda CTZ: F.one_hot(CTZ.argmax(dim=1), num_classes=CTZ.size(1)).float().detach()
        else:
            raise ValueError('Invalid q_generator')

        self.use_whole_batch_as_positives = use_whole_batch_as_positives

        self.model = self.init_model()
        if self.adapt_exclusively is not None and not any(name in param_name for param_name in self.model.state_dict().keys() for name in self.adapt_exclusively):
            raise ValueError(f'The given model does not contain any layers matching any of {self.adapt_exclusively}')

        # metrics
        self.pre_adaption_acc = Accuracy(compute_on_step=False)
        for i in range(self.num_steps):
            setattr(self, f'after_{i+1}_adap_steps_acc', Accuracy(compute_on_step=False))

    def naively_normalized(self, CTZ):
        cos_sim_pos = CTZ + 1 # all values between 0 and 2
        return (cos_sim_pos / torch.sum(cos_sim_pos, dim=1).unsqueeze(1)).detach()

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

        return backbone(
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

    def on_train_start(self):
        # if batch_norm is used, freeze running mean and var,
        # to make sure any improvements can be attributed exclusively to the adaption, not to better running stats
        if self.norm_layer == 'batch_norm':
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.track_running_stats = False # stops updating running stats, but keeps the ones it has accumulated so far

        # saving the original model to reset self.model to after every image prediction
        self.original_model = copy.deepcopy(self.model)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if batch_idx % self.num_steps == 0 and batch_idx != 0:
            # new test time adaption starts: reset model
            self.model.load_state_dict(self.original_model.state_dict())

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        # last element of inputs: batch used for supervised training
        # rest of the inputs: multicrop-batches for SwAV
        if len(inputs) != sum(self.nmb_crops) + 1:
            raise ValueError(
                f'Got inputs of invalid length {len(inputs)}, when length {sum(self.nmb_crops) + 1} was expected.'
            )

        swav_loss = 0

        ## SwAV

        # 1. normalize the prototypes
        if not self.freeze_prototypes:
            with torch.no_grad():
                w = self.model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.model.prototypes.weight.copy_(w)

        # 2. multi-res forward passes
        if batch_idx % self.num_steps == 0:
            # first step of adaption: also pass supervised batch through, to update the pre-adaption accuracy
            logits, (embedding, output) = self.model(inputs)
            logits = logits.detach() # the resulting loss doesn't get passed to the optimizer anyway, but just to be sure
        else:
            # swav only
            embedding, output = self.model.forward_swav(inputs[:-1])

        if self.loss_type == 'nearest_prototype_distance':
            output = output.detach()
        else:
            embedding = embedding.detach()
        bs = inputs[0].size(0)

        # 3. swav loss computation
        for i, crop_id in enumerate(self.crops_for_assign):

            if self.loss_type == 'nearest_prototype_distance':
                nearest_prototypes_idx = output[bs * crop_id:bs * (crop_id + 1)].argmax(dim=1)
                swav_loss += torch.mean(torch.linalg.vector_norm(embedding[bs * crop_id:bs * (crop_id + 1)] - self.model.prototypes.weight[nearest_prototypes_idx], dim=1))
            else:
                with torch.no_grad():
                    out = output[bs * crop_id:bs * (crop_id + 1)]

                    # 5. get assignments
                    q = self.get_q(out)

                if self.use_whole_batch_as_positives:
                    # cluster assignment prediction
                    p = self.softmax(output / self.temperature) # generate the p-term in the swav loss for all crops of all images in the batch

                    # matrix of size (bs, sum(self.nmb_crops) * bs), which contains the potential losses for the prediction of all the cosine_similarites by all the assignments in the batch (for this crop)
                    potential_sublosses = -torch.mm(q, torch.log(p).t())

                    # the value at position i,j of this matrix defines, whether the i'th soft assignment (q) should be matched with the j'th cosine_similarity
                    # i is in [0, bs), while j is in [0, sum(self.nmb_crops) * bs)
                    # because assignments are only made for one crop per loop iteration, and cosine_similarities (`output`/`p`) are available for all crops at the same time
                    classes_matching_index_not_matching = torch.ones_like(potential_sublosses).bool() # all true, since all images of the batch are in fact the same image (and therefore have the same class)

                    # don't match to the exact same image itself (note that this only excludes this exact crop of the image, not the other crops of multi-crop)
                    for i in range(bs):
                        classes_matching_index_not_matching[i, crop_id*bs + i] = False

                    swav_loss += torch.mean(potential_sublosses[classes_matching_index_not_matching])
                else:
                    # cluster assignment prediction
                    subloss = 0
                    for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                        p = self.softmax(output[bs * v:bs * (v + 1)] / self.temperature)
                        subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
                    swav_loss += subloss / (np.sum(self.nmb_crops) - 1)

        swav_loss /= len(self.crops_for_assign)

        if batch_idx % self.num_steps == 0:
            ce_loss = F.cross_entropy(logits[:1], labels[:1])
            self.pre_adaption_acc(logits[:1], labels[:1])

            joint_loss = swav_loss + self.supervised_weight * ce_loss

            self.log('pre_adap_ce_loss', ce_loss, on_step=False, on_epoch=True)
            self.log('pre_adap_swav_loss', swav_loss, on_step=False, on_epoch=True)
            self.log('pre_adap_joint_loss', joint_loss, on_step=False, on_epoch=True)

        return swav_loss

    def on_after_backward(self):
        if self.freeze_prototypes:
            for name, p in self.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

        if self.freeze_norm_layers:
            for name, p in self.model.named_parameters():
                if "bn" in name:
                    p.grad = None

        if self.adapt_exclusively is not None:
            for param_name, p in self.model.named_parameters():
                if not any(name in param_name for name in self.adapt_exclusively) and not 'prototypes' in param_name:
                    p.grad = None


    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # extract only first image of non-augmented batch (all images are identical)
        image = batch[0][-1][:1]
        label = batch[1][:1]
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image)
        # after n-th adaption step: update n-th accuracy
        getattr(self, f'after_{(batch_idx % self.num_steps) + 1}_adap_steps_acc')(logits, label)

    def on_train_epoch_end(self):
        self.log('pre_adap_acc', self.pre_adaption_acc)
        for i in range(self.num_steps):
            self.log(f'after_{i+1}_adap_steps_acc', getattr(self, f'after_{i+1}_adap_steps_acc'))

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{'params': params, 'weight_decay': weight_decay}, {'params': excluded_params, 'weight_decay': 0.}]

    def configure_optimizers(self):
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

        return optimizer

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

        # * new arguments
        parser.add_argument('--corruption_type', type=str, default='none', help='type of corruption to evaluate')
        parser.add_argument('--severity', type=int, default=5, help='severity of corruption')
        parser.add_argument('--freeze_prototypes', type=true_false, action='store', default=True,
            help='whether to keep the prototypes frozen during evaluation'
        )
        parser.add_argument('--freeze_norm_layers', type=true_false, action='store', default=False,
            help='whether to keep the norm layers frozen during evaluation'
        )
        parser.add_argument("--num_steps", default=1, type=int, help="number of adaption steps per image")
        parser.add_argument('--q_generator', type=str, default='softmax_normalized', choices=['sinkhorn', 'softmax_normalized', 'naively_normalized', 'hard_assignment'],
            help=('how to generate the matrix Q from C^T*Z (see https://arxiv.org/abs/2006.09882 for context): '
                  '"softmax_normalized" (default) normalizes the columns to be positive and sum to 1 using a similar calculation as sinkhorn, '
                  '"sinkhorn" uses the sinkhorn algorithm as in SwAV, '
                  '"naively_normalized" adds 1 to C^T*Z (which are cosine similarity values) to make all values positive, then normalizes the columns to sum to 1, '
                  '"hard_assignment" takes the argmax of the columns of C^T*Z and generates a one-hot vector')
        )
        parser.add_argument("--lr_scaling", default=1., type=float,
            help="Factor to scale the learning rate with. If learning_rate itself is specified, this parameter will be ignored."
        )
        parser.add_argument("--adapt_exclusively", type=str, default='all',
            choices=['all', 'bn', 'layer1', 'layer2', 'layer3', 'layer4', 'shared_head', 'from_layer2', 'from_layer3', 'from_layer4'],
            help=("Specify one part of the network to that gets adapted, the rest of the network will be frozen."
                  'Note that "bn" adapts all norm layers in the network, regardless of the type of norm layer that is used.'
                  'By default, the entire network gets adapted.')
        )
        parser.add_argument("--loss_type", default='swav', type=str, choices=['swav', 'nearest_prototype_distance'])
        parser.add_argument("--use_whole_batch_as_positives", default=False, type=true_false)

        parser.add_argument("--test", default=False, type=true_false,
            help=('When using the office31 dataset, the data being used here is the same as the validation data during training. '
                  'This flag changes it to completely new data to test with (after potential hyperparameter optimization on the validation data). '
                  'For other datasets, this has no effect.')
        )

        # * hardware/filesystem-specific arguments: can be loaded from checkpoint file, but should be overwritten in case of a different setup
        parser.add_argument("--data_dir", type=str, help="path to download data")
        parser.add_argument("--gpus", type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", type=int, help="num of workers per GPU")
        parser.add_argument("--num_nodes", type=int, help="number of nodes for training")

        # * can be loaded from checkpoint file, but should probably be overwritten
        parser.add_argument("--batch_size", type=int, help="batch size to use for test-time adaption for one image")

        # * will be loaded from checkpoint file, but can be overwritten (no default value so by default it doesn't overwrite)
        parser.add_argument("--dataset", type=str, help="cifar10, cifar100, imagenet, office31")
        parser.add_argument("--optimizer", type=str, help="choose between adam/lars/sgd")
        parser.add_argument("--gaussian_blur", type=true_false, action='store', help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, help="jitter strength")
        parser.add_argument("--nmb_crops", type=list_from_string(int), help="list of number of crops (example: [2, 6])")
        parser.add_argument("--size_crops", type=list_from_string(int), help="crops resolutions (example: [224, 96])")
        parser.add_argument(
            "--min_scale_crops",
            type=list_from_string(float),
            help="argument in RandomResizedCrop (example: [0.14, 0.05])"
        )
        parser.add_argument(
            "--max_scale_crops",
            type=list_from_string(float),
            help="argument in RandomResizedCrop (example: [1., 0.14])"
        )
        parser.add_argument("--weight_decay", type=float, help="weight decay")
        parser.add_argument("--learning_rate", type=float, help="base learning rate")
        parser.add_argument(
            "--crops_for_assign",
            type=list_from_string(int),
            help="list of crops id used for computing assignments"
        )
        parser.add_argument("--temperature", type=float, help="temperature parameter in training loss")
        parser.add_argument("--epsilon", type=float, help="regularization parameter for Sinkhorn-Knopp algorithm")
        parser.add_argument("--sinkhorn_iterations", type=int, help="number of iterations in Sinkhorn-Knopp algorithm")

        # * will only be loaded from the checkpoint file
        # parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
        # parser.add_argument("--nmb_prototypes", default=512, type=int, help="number of prototypes")
        # parser.add_argument('--supervised_hidden_mlp', type=int, default=2048, help='hidden layer of non-linear supervised learning head')
        # parser.add_argument("--first_conv", action='store_false')
        # parser.add_argument("--maxpool1", action='store_false')
        # parser.add_argument("--norm_layer", type=str, default='batch_norm', action='store', help='batch_norm, group_norm')
        # parser.add_argument("--num_groups", type=int, default=16, action='store',
        #     help='in case group_norm is chosen as norm_layer, this sets the number of groups'
        # )
        # parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        # parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        # parser.add_argument("--fp32", action='store_true')

        return parser


def cli_main():
    from CifarCorrupted import CIFAR10CDataModule, CIFAR100CDataModule
    from ImageNetCorrupted import ImageNetCDataModule
    from Office31 import Office31DataModule
    from pl_bolts.models.self_supervised.swav.transforms import SwAVEvalDataTransform, SwAVTrainDataTransform
    from pytorch_lightning import loggers as pl_loggers
    from torchvision import transforms as transforms

    parser = ArgumentParser()

    parser.add_argument('--wandb_project', default='uncategorized', type=str, help='name of the wandb project to save this run in')
    parser.add_argument('--wandb_log_dir', default='wandb_logs', type=str, help='directory to save wandb logs in')

    parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint to load the model from')
    parser.add_argument('--artifact_name', type=str, action='store',
        help='wandb artifact name to load the model from (provide either this or checkpoint_path)'
    )
    parser.add_argument('--artifact_dir', type=str, action='store', default='wandb_artifacts', help='directory to save wandb artifacts in')
    parser.add_argument('--wandb_entity', type=str, help='name of the wandb entity (account) to load an artifact from')
    parser.add_argument('--logger_prefix', default='test_time_adaption', type=str,
        help='prefix to the name of all the logged metrics (empty string for no prefix)',
    )

    parser.add_argument("--fast_dev_run", default=1, type=int)

    # model args
    parser = SupervisedSwAVTTAEvaluator.add_model_specific_args(parser)
    args = parser.parse_args()

    assert (args.checkpoint_path is not None) ^ (args.artifact_name is not None), \
        'EITHER a checkpoint path OR an artifact name has to be provided'

    if args.checkpoint_path is not None:
        assert os.path.isfile(args.checkpoint_path), 'The specified checkpoint file does not exist'

    if args.learning_rate is not None:
        args.lr_scaling = 1

    logger = pl_loggers.WandbLogger(
        save_dir=args.wandb_log_dir,
        project=args.wandb_project,
        prefix=args.logger_prefix,
        log_model=False,
    )
    logger.LOGGER_JOIN_CHAR = '/'

    if args.artifact_name is not None:
        assert args.wandb_entity is not None, "Need to specify a wandb_entity to load an artifact"
        artifact_dirname = args.artifact_name.replace(":", "-")

        if type(logger.experiment) is not pl_loggers.base.DummyExperiment:
            artifact = logger.experiment.use_artifact(f'{args.wandb_entity}/{args.wandb_project}/{args.artifact_name}', type='model')
            if not os.path.isdir(os.path.join(args.artifact_dir, artifact_dirname)):
                # artifact has not been downloaded yet -> download artifact and store its path in hyperparameter_checkpoint
                args.checkpoint_path = os.path.join(
                    artifact.download(os.path.join(
                        args.artifact_dir,
                        artifact_dirname
                    )),
                    'model.ckpt'
                )
            else:
                # artifact is already downloaded, because it was used before -> just store its path in hyperparameter_checkpoint
                args.checkpoint_path = os.path.join(
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
            args.checkpoint_path = os.path.join(
                args.artifact_dir,
                artifact_dirname,
                'model.ckpt'
            )

    # load hyperparameters from the checkpoint
    while True:
        try:
            hyperparameters = load_checkpoint(args.checkpoint_path)['hyper_parameters']
            break
        except:
            print('Waiting for download...')
            sleep(1)

    hyperparameters['source_domain_dataset'] = hyperparameters['dataset']

    if hyperparameters['dataset'].startswith('office31') and args.dataset in (None, hyperparameters['dataset']):
        raise ValueError(f'Trying to do test-time adaption on the domain the model was trained on ({hyperparameters["dataset"][9:]})')

    # overwrite hyperparameters that were explicitly set using commandline arguments, and add new ones
    hyperparameters.update({k: v for k, v in args.__dict__.items() if v is not None})

    # write the updated hyperparameters back into args' attributes for nicer access
    args.__dict__ = hyperparameters

    if args.corruption_type == 'none':
        args.corruption_type = args.dataset

    if args.adapt_exclusively == 'from_layer2':
        args.adapt_exclusively = ['layer2', 'layer3', 'layer4', 'shared_head', 'projection_head']
    if args.adapt_exclusively == 'from_layer3':
        args.adapt_exclusively = ['layer3', 'layer4', 'shared_head', 'projection_head']
    if args.adapt_exclusively == 'from_layer4':
        args.adapt_exclusively = ['layer4', 'shared_head', 'projection_head']

    args.learning_rate *= args.lr_scaling

    logger.experiment.name = '/'.join([
        args.dataset,
        'tta',
        args.artifact_name if 'artifact_name' in args.__dict__ else os.path.basename(args.checkpoint_path)] +
        ([args.corruption_type, str(args.severity)] if args.corruption_type != args.dataset else [])
    )

    if args.dataset == 'cifar10':
        dm = CIFAR10CDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption_type=args.corruption_type,
            severity=args.severity,
            num_repeats=args.batch_size * args.num_steps
        )

        normalization = cifar10_normalization()
    elif args.dataset == 'cifar100':
        dm = CIFAR100CDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption_type=args.corruption_type,
            severity=args.severity,
            num_repeats=args.batch_size * args.num_steps
        )
        normalization = cifar10_normalization()
    elif args.dataset.startswith('office31'):
        if len(args.dataset) < 9 or args.dataset[8] != ':':
            raise ValueError('Violated expected format for office31 dataset name: "office31:<domain>", e.g."office31:amazon"')

        dm = Office31DataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            domain=args.dataset[9:],
            num_repeats=args.batch_size * args.num_steps,
            val_split=0.2 # creating the same split as during training
        )
        if args.test:
            dm.train_dataloader = dm.val_dataloader

        normalization = imagenet_normalization()
    elif args.dataset == 'imagenet':
        dm = ImageNetCDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption_type=args.corruption_type,
            severity=args.severity,
            num_repeats=args.batch_size * args.num_steps
        )

        normalization = imagenet_normalization()
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    assert not args.supervised_only, "Test-Time Adaption isn't possible with a model trained in a purely supervised manner"

    dm.train_transforms = dm.val_transforms = SwAVTrainDataTransform(
        normalize=normalization,
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength
    )

    dm.train_transforms.transform[-1] = dm.val_transforms.transform[-1] = transforms.Compose([
        transforms.ToTensor(),
        normalization,
    ])

    # model init
    model = SupervisedSwAVTTAEvaluator.load_from_checkpoint(**args.__dict__)

    trainer = pl.Trainer(
        max_epochs=1, # one epoch corresponds with one complete validation
        checkpoint_callback=False,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        distributed_backend='ddp' if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        fast_dev_run=args.fast_dev_run,
        logger=logger,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
