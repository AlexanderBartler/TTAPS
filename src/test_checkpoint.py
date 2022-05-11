import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
)

from CifarCorrupted import CIFAR10CDataModule, CIFAR100CDataModule
from Office31 import Office31DataModule
from ImageNetCorrupted import ImageNetCDataModule
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms as transforms

from supervised_swav_module import SupervisedSwAV

def main():
    parser = ArgumentParser()

    parser.add_argument('--wandb_project', default='uncategorized', type=str, help='name of the wandb project to save this run in')
    parser.add_argument('--wandb_log_dir', default='wandb_logs', type=str, help='directory to save wandb logs in')

    parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint to load the model from')
    parser.add_argument('--artifact_name', type=str, action='store',
        help='wandb artifact name to load the model from (provide either this or checkpoint_path)'
    )
    parser.add_argument('--artifact_dir', type=str, action='store', default='wandb_artifacts', help='directory to save wandb artifacts in')
    parser.add_argument('--wandb_entity', type=str, help='name of the wandb entity (account) to load an artifact from')
    parser.add_argument('--logger_prefix', default='supervised_swav', type=str,
        help='prefix to the name of all the logged metrics (empty string for no prefix)',
    )

    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, imagenet')
    parser.add_argument('--corruption_type', type=str, default='none', help='type of corruption to evaluate')
    parser.add_argument('--severity', type=int, default=5, help='severity of corruption')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--data_dir", type=str, default=".", help="path to download data")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument("--fp32", action='store_true')

    parser.add_argument("--test", action='store_true',
        help=('When using the office31 dataset, the data being used here is the same as the validation data during training. '
              'This flag changes it to completely new data to test with (after potential hyperparameter optimization on the validation data). '
              'For other datasets, this has no effect.')
    )

    args = parser.parse_args()

    assert (args.checkpoint_path is not None) ^ (args.artifact_name is not None), \
        'EITHER a checkpoint path OR an artifact name has to be provided'

    if args.checkpoint_path is not None:
        assert os.path.isfile(args.checkpoint_path), 'The specified checkpoint file does not exist'

    logger = pl_loggers.WandbLogger(
        save_dir=args.wandb_log_dir,
        project=args.wandb_project,
        prefix=args.logger_prefix,
    )
    logger.LOGGER_JOIN_CHAR = '/'

    if args.corruption_type == 'none':
        args.corruption_type = args.dataset

    if args.artifact_name is not None:
        assert args.wandb_entity is not None, "Need to specify a wandb_entity to load an artifact"
        artifact_dirname = args.artifact_name.replace(":", "-")

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

    logger.experiment.name = '/'.join([
        args.dataset,
        'test',
        args.artifact_name or os.path.basename(args.checkpoint_path)] +
        ([args.corruption_type, str(args.severity)] if args.corruption_type != args.dataset else [])
    )

    if args.dataset == 'cifar10':
        dm = CIFAR10CDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption_type=args.corruption_type,
            severity=args.severity,
        )

        normalization = cifar10_normalization()
    elif args.dataset == 'cifar100':
        dm = CIFAR100CDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption_type=args.corruption_type,
            severity=args.severity,
        )
        normalization = cifar10_normalization()
    elif args.dataset == 'imagenet':
        dm = ImageNetCDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption_type=args.corruption_type,
            severity=args.severity,
        )

        normalization = imagenet_normalization()
    elif args.dataset.startswith('office31'):
        if len(args.dataset) < 9 or args.dataset[8] != ':':
            raise ValueError('Violated expected format for office31 dataset name: "office31:<domain>", e.g."office31:amazon"')

        dm = Office31DataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            domain=args.dataset[9:],
            val_split=0.2 # creating the same split as during training
        )
        if args.test:
            dm.train_dataloader = dm.val_dataloader

        normalization = imagenet_normalization()
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalization,
    ])

    if args.dataset.startswith('office31'):
        dm.setup(stage='fit')
    else:
        dm.setup()

    trainer = pl.Trainer(
        gpus=1,
        num_nodes=1,
        precision=32 if args.fp32 else 16,
        logger=logger,
    )

    model = SupervisedSwAV.load_from_checkpoint(args.checkpoint_path)

    trainer.test(model=model, test_dataloaders=dm.train_dataloader())

if __name__ == '__main__':
    main()
