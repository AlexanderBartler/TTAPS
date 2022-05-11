from pl_bolts.datamodules import CIFAR10DataModule
from torchvision.datasets import CIFAR100

class CIFAR100DataModule(CIFAR10DataModule):

    name = "cifar100"
    dataset_cls = CIFAR100

    @property
    def num_classes(self) -> int:
        """
        Return:
            100
        """
        return 100
