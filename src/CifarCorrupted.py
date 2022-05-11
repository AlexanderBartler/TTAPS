"""
Cifar10/100 Corrupted Lightning data modules
"""
import os
from PIL import Image
import pytorch_lightning as pl
from typing import  Optional, Sized, Any, Callable, Tuple
from torch.utils.data import  DataLoader
# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data.sampler import Sampler
import numpy as np
from torchvision.datasets.vision import VisionDataset

# pip install git+https://github.com/RobustBench/robustbench.git@v0.2.1
from robustbench.data import load_cifar10c, load_cifar100c


class SequentialRepeatSampler(Sampler[int]):
    r"""Samples and repeats elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source, num_repeats=1):
        self.data_source = data_source
        self.num_repeats = num_repeats

    def __iter__(self):
        # return iter(range(len(self.data_source)))
        # return iter with repetition of index
        return iter([i for i in range(len(self.data_source)) for _ in range(self.num_repeats)])

    def __len__(self) -> int:
        return len(self.data_source) * self.num_repeats


class CIFAR10C(VisionDataset):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            corruption='fog',
            severity=5,
            download: bool = False,
    ) -> None:

        super(CIFAR10C, self).__init__(root, transform=transform,
                                       target_transform=target_transform)

        self.corruption = [corruption]  # has to be a list since robust bench loader needs it
        self.severity = severity
        x_test, y_test = load_cifar10c(n_examples=10000, corruptions=self.corruption, severity=self.severity,
                                       data_dir=root)

        # as in the original cifar10, data: ndarry  uint8 (num_samples, 3, 32,32), label: list (int)
        self.data = np.array(255 * x_test.transpose(1, 3).transpose(1,2)).astype(np.uint8)
        self.targets = y_test.tolist()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class CIFAR100C(VisionDataset):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            corruption='fog',
            severity=5,
            download: bool = False,
    ) -> None:

        super(CIFAR100C, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.corruption = [corruption]  # has to be a list since robust bench loader needs it
        self.severity = severity
        x_test, y_test = load_cifar100c(n_examples=10000, corruptions=self.corruption, severity=self.severity,
                                        data_dir=root)

        # as in the original cifar10, data: ndarry  uint8 (num_samples, 3, 32,32), label: list (int)
        self.data = np.array(255 * x_test.transpose(1, 3).transpose(1,2)).astype(np.uint8)
        self.targets = y_test.tolist()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class CIFAR10CDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/path/to/data", batch_size: int = 32, num_workers=0,
                 corruption_type: str = 'cifar10', severity=5,
                 num_repeats=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_repeats = num_repeats
        self.corruption_type = corruption_type
        self.severity = severity

    def setup(self, stage: Optional[str] = None):
        if self.corruption_type != 'cifar10':
            # get corrupted data
            self.data = CIFAR10C(self.data_dir, transform=self.train_transforms, corruption=self.corruption_type,
                                 severity=self.severity)
        else:
            # get normal cifar10 val data
            self.data = CIFAR10(self.data_dir, train=False, download=True, transform=self.train_transforms)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=False,
                          sampler=SequentialRepeatSampler(self.data, num_repeats=self.num_repeats),
                          num_workers=self.num_workers)


class CIFAR100CDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/path/to/data", batch_size: int = 32, num_workers=0,
                 corruption_type: str = 'cifar100', severity=5,
                 num_repeats=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_repeats = num_repeats
        self.corruption_type = corruption_type
        self.severity = severity

    def setup(self, stage: Optional[str] = None):
        if self.corruption_type != 'cifar100':
            # get corrupted data
            self.data = CIFAR100C(self.data_dir, transform=self.train_transforms, corruption=self.corruption_type,
                                  severity=self.severity)
        else:
            # get normal cifar10 val data
            self.data = CIFAR100(self.data_dir, train=False, download=True, transform=self.train_transforms)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=False,
                          sampler=SequentialRepeatSampler(self.data, num_repeats=self.num_repeats),
                          num_workers=self.num_workers)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # Perform dataset testing
    print("Start Testing dataset loading...")
    batch_size = 16  # for testing should be x^2, e.g 2^2 = 4
    num_repeats = 8
    dm = CIFAR10CDataModule(data_dir="/data/public/", batch_size=batch_size, corruption_type='snow', severity=5,
                            num_repeats=num_repeats, num_workers=4)
    dm.train_transforms = transforms.Compose([transforms.ToTensor()])
    dm.setup()
    test = dm.train_dataloader()

    # check if dataset is fully parsable
    print("Parse complete dataset, may take a while...")
    for image, label in test:
        pass

    # Shape and visual check
    print("Check shapes and plot a batch")
    for image, label in test:
        # Check shapes
        assert image.shape[0] == batch_size, "Batch Size does not match"
        assert image.shape[1] == 3, "Number of channels does not match"
        assert image.shape[2] == 32, "Image width/height does not match"
        # Visually check image
        plt.figure(figsize=(12, 12))
        for k in range(batch_size):
            plt.subplot(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), k + 1)
            plt.imshow(image[k, :, :, :].permute(1, 2, 0))
            plt.axis('off')
            plt.title(f"Label: {label[k]}")
        plt.show()
        break
