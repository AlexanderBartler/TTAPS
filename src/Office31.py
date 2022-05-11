"""
Office31 Lightning data module
"""
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from PIL import Image
from typing import Optional, Sized, Any, Callable, Tuple
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
from typing import Union

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


class Office31(VisionDataset):

    labels = {
        'back_pack': 0,
        'bike': 1,
        'bike_helmet': 2,
        'bookcase': 3,
        'bottle': 4,
        'calculator': 5,
        'desk_chair': 6,
        'desk_lamp': 7,
        'desktop_computer': 8,
        'file_cabinet': 9,
        'headphones': 10,
        'keyboard': 11,
        'laptop_computer': 12,
        'letter_tray': 13,
        'mobile_phone': 14,
        'monitor': 15,
        'mouse': 16,
        'mug': 17,
        'paper_notebook': 18,
        'pen': 19,
        'phone': 20,
        'printer': 21,
        'projector': 22,
        'punchers': 23,
        'ring_binder': 24,
        'ruler': 25,
        'scissors': 26,
        'speaker': 27,
        'stapler': 28,
        'tape_dispenser': 29,
        'trash_can': 30,
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            domain='amazon', # amazon, dslr, webcam
            download: bool = False,
    ) -> None:

        super(Office31, self).__init__(root, transform=transform,
                                       target_transform=target_transform)

        if domain not in ('amazon', 'dslr', 'webcam'):
            raise ValueError(f'Invalid domain for office31: "{domain}"')

        root_path = Path(root)
        if not (root_path / 'office31').exists():
            raise ValueError(f"Office31 dataset can't be found at {root_path / 'office31'}")

        if download:
            print('Download is not supported for Office31, but files seem to be present.')
            return

        if not train:
            raise ValueError('For Office31, no test set is provided')

        self.data_targets = []

        for file in sorted((root_path / 'office31' / domain / 'images').glob('*/*.jpg')):
            self.data_targets.append((np.array(Image.open(file)), self.labels[file.parent.name]))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # preprocessing step which is always applied
        img = transforms.Resize((256, 256))(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data_targets)


class Office31DataModule(VisionDataModule):

    name = 'office31'
    EXTRA_ARGS = {}
    dataset_cls = Office31
    num_classes = 31
    labels = dataset_cls.labels

    def __init__(self, data_dir: str = "/path/to/data", batch_size: int = 32, shuffle: bool = False,
                 num_workers: int = 0, val_split: Union[int, float] = 0.2, domain: str ='amazon', num_repeats: int = 1):
        num_images = {
            'amazon': 2817,
            'dslr': 498,
            'webcam': 795,
        }[domain]
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            val_split=val_split,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=num_images % (batch_size*num_repeats) == 1,
        )
        self.num_repeats = num_repeats
        # NOTE: using "self.EXTRA_ARGS['domain'] = domain" would change the class attribute, setting the domain for all instances of this class!
        # this instead shadows the class attribute by creating an instance attribute of the same name (which takes precedence)
        self.EXTRA_ARGS = {'domain': domain}
        self.num_samples = self._get_splits(num_images)[0]

    # override
    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        if self.num_repeats > 1:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                sampler=SequentialRepeatSampler(dataset, num_repeats=self.num_repeats),
            )
        else:
            return super()._data_loader(dataset, shuffle)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # Perform dataset testing
    print("Start Testing dataset loading...")
    batch_size = 16  # for testing should be x^2, e.g 2^2 = 4
    num_repeats = 8

    # run checks for all sets of all domains
    for domain in ('amazon', 'dslr', 'webcam'):
        print(f'Testing domain "{domain}"...')
        dm = Office31DataModule(data_dir="D:/pytorch_datasets", batch_size=batch_size, num_repeats=num_repeats, num_workers=1, domain=domain, val_split=0)
        dm.train_transforms = dm.val_transforms = transforms.ToTensor()
        dm.setup(stage='fit') # stage='fit' to avoid creating test set

        for idx, loader in enumerate((dm.train_dataloader(), dm.val_dataloader())):
            if len(loader) == 0:
                continue
            print(('Training', 'Validation')[idx], 'Set')
            # check if dataset is fully parsable
            print("Parse complete dataset, may take a while...")
            for image, label in loader:
                pass

            # Shape and visual check
            print("Check shapes and plot a batch")
            image, label = next(iter(loader))
            # Check shapes
            assert image.shape[0] == min(batch_size, len(loader)*num_repeats), "Batch Size does not match"
            assert image.shape[1] == 3, "Number of channels does not match"
            assert image.shape[2] == 256, "Image width/height does not match"
            # Visually check image
            plt.figure(figsize=(12, 12))
            for k in range(image.shape[0]):
                plt.subplot(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), k + 1)
                plt.imshow(image[k, :, :, :].permute(1, 2, 0))
                plt.axis('off')
                plt.title(f"Label: {label[k]}")
            plt.show()
