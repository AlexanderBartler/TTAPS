import os
from PIL import Image
import pytorch_lightning as pl
from typing import Optional, Sized, Any, Callable, Tuple
from torch.utils.data import DataLoader
# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder
from torch.utils.data.sampler import Sampler
import numpy as np
from torchvision.datasets.vision import VisionDataset

# pip install git+https://github.com/RobustBench/robustbench.git@v0.2.1
from robustbench.data import load_cifar10c, load_cifar100c

import hashlib
import shutil
from pathlib import Path
from typing import Set

import requests
from tqdm import tqdm

ZENODO_ENTRY_POINT = "https://zenodo.org/api"
RECORDS_ENTRY_POINT = f"{ZENODO_ENTRY_POINT}/records/"
ZENODO_ID = 2235448
ZENODO_FILENAMES = {'blur.tar', 'digital.tar', 'extra.tar', 'noise.tar', 'weather.tar'}
CHUNK_SIZE = 65536


class DownloadError(Exception):
    pass


def download_file(url: str, save_dir: Path, total_bytes: int) -> Path:
    """Downloads large files from the given URL.

    From: https://stackoverflow.com/a/16696317

    :param url: The URL of the file.
    :param save_dir: The directory where the file should be saved.
    :param total_bytes: The total bytes of the file.
    :return: The path to the downloaded file.
    """
    local_filename = save_dir / url.split('/')[-1]
    print(f"Starting download from {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            iters = total_bytes // CHUNK_SIZE
            for chunk in tqdm(r.iter_content(chunk_size=CHUNK_SIZE),
                              total=iters):
                f.write(chunk)

    return local_filename


def file_md5(filename: Path) -> str:
    """Computes the MD5 hash of a given file"""
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(32768), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def zenodo_download(record_id: str, filenames_to_download: Set[str],
                    save_dir: str) -> None:
    """Downloads the given files from the given Zenodo record.

    :param record_id: The ID of the record.
    :param filenames_to_download: The files to download from the record.
    :param save_dir: The directory where the files should be saved.
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    url = f"{RECORDS_ENTRY_POINT}/{record_id}"
    res = requests.get(url)
    files = res.json()["files"]
    files_to_download = list(
        filter(lambda file: file["key"] in filenames_to_download, files))

    for file in files_to_download:
        if (save_dir / file["key"]).exists():
            continue
        file_url = file["links"]["self"]
        file_checksum = file["checksum"].split(":")[-1]
        filename = download_file(file_url, save_dir, file["size"])
        if file_md5(filename) != file_checksum:
            raise DownloadError(
                "The hash of the downloaded file does not match"
                " the expected one.")
        print("Download finished, extracting...")
        shutil.unpack_archive(filename,
                              extract_dir=save_dir,
                              format=file["type"])
        print("Downloaded and extracted.")


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


class ImageNetCDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/path/to/data", batch_size: int = 32, num_workers=0,
                 corruption_type: str = 'imagenet', severity=5,
                 num_repeats=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_repeats = num_repeats
        self.corruption_type = corruption_type
        self.severity = severity

    def setup(self, stage: Optional[str] = None):
        if self.corruption_type != 'imagenet':
            # get corrupted data
            # self.data = CIFAR10C(self.data_dir, transform=self.train_transforms, corruption=self.corruption_type,
            #                     severity=self.severity)
            self.data_dir = os.path.join(self.data_dir, 'IMAGENET-C')
            if not os.path.exists(self.data_dir):
                # download and extract dataset
                zenodo_download(ZENODO_ID, ZENODO_FILENAMES, self.data_dir)
                assert "Please download and extract dataset"
            self.corruption_dir = os.path.join(self.data_dir, self.corruption_type, str(self.severity))
            self.data = ImageFolder(self.corruption_dir, transform=self.train_transforms)
            assert NotImplementedError
        else:
            # get normal cifar10 val data
            self.data_dir = os.path.join(self.data_dir, 'imagenet2012', 'val')
            self.data = ImageFolder(self.data_dir, transform=self.train_transforms)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=False,
                          sampler=SequentialRepeatSampler(self.data, num_repeats=self.num_repeats),
                          num_workers=self.num_workers)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # Perform dataset testing
    print("Start Testing dataset loading...")
    batch_size = 4  # for testing should be x^2, e.g 2^2 = 4
    num_repeats = 1
    corruption = 'imagenet'
    dm = ImageNetCDataModule(data_dir="/data/public/", batch_size=batch_size, corruption_type=corruption, severity=5,
                             num_repeats=num_repeats, num_workers=4)
    # for original imagenet
    if corruption == 'imagenet':
        dm.train_transforms = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224), transforms.ToTensor()])
    # for corrupted imagenet
    else:
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
        assert image.shape[2] == 224, "Image width/height does not match"
        # Visually check image
        plt.figure(figsize=(12, 12))
        for k in range(batch_size):
            plt.subplot(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), k + 1)
            plt.imshow(image[k, :, :, :].permute(1, 2, 0))
            plt.axis('off')
            plt.title(f"Label: {label[k]}")
        plt.show()
        break
