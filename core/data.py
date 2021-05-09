"""
Custom dataset class for data manipulation
"""


from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import torchvision
from torchvision.utils import save_image


class CustomImage(Dataset):
    def __init__(self, dir: str, cla: list = None,
                 trans: torchvision.transforms = None) -> object:
        """
        :rtype: object
        """
        super(CustomImage, self).__init__()

        self.dir = dir
        if self.dir:
            self.cla = cla

        if not trans:
            self.trans = torchvision.transforms.ToTensor()
        else:
            self.trans = trans

        classes, class_to_idx = self._find_class()

        self.sampled = self._make_dataset(self.dir, class_to_idx)
        self.tester()

    def _find_class(self) -> tuple:
        """
        :finds the class names in a given folder:
        :return classes, target:
        """
        if self.cla:
            classes = [d.name for d in os.scandir(self.dir) if d.is_dir() and d.name in self.cla]
        else:
            classes = [d.name for d in os.scandir(self.dir) if d.is_dir()]

        classes.sort()
        class_to_xid = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_xid

    def _augment_data(self, trans: torchvision.transforms,
                      num_samples: int, process: str = None):
        """
        Args:
            trans
            num_samples
        Returns:
            None
        """
        count = 0
        data = self.sampled

        for i in range(num_samples):
            for x, _ in data:
                image = Image.open(x)

                x = x.split('.')
                x = '.' + '.' + str(x[0]) + str(x[1]) + str(x[2]) + str(count) + str(process) + '.' + str(x[3])
                image = trans(image)

                save_image(image, x)
                count += 1

    @staticmethod
    def _make_dataset(dir, class_to_idx) -> tuple:
        """
        Args:
            class_to_xid dict with classes

        Returns:
            target
            path list
        """
        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)
        return instances

    def tester(self):
        """

        :return None:
        """
        for i, j in self.sampled:
            try:
                Image.open(i)
            except Exception as e:
                print(e)
                # os.remove(i)

    def __len__(self) -> int:
        """
        Returns:
            Length of dataset
        """
        return len(self.sampled)

    def __getitem__(self, xid) -> tuple:
        """
        Args:
            xid

        Returns:
            Image
            Label
        """
        image_path, label = self.sampled[xid]
        image = self.trans(Image.open(image_path))
        return image, label


def splitdata(data: CustomImage, splits: list,
              batch_size: int) -> torch.utils.data.DataLoader:
    """
    Args:
        Object of class CustomImage
        splits
        batch_size

    Returns:
        train_loader
        test_loader
    """

    train, test = torch.utils.data.random_split(data, splits)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True,
                                              num_workers=2)
    return train_loader, test_loader


if __name__ == '__main__':
    data = CustomImage(dir='../train', cla=['Sample001', 'Sample002', 'Sample003',
                                            'Sample004', 'Sample005', 'Sample006',
                                            'Sample007', 'Sample008', 'Sample009',
                                            'Sample010'])
    aug_trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomPerspective(),
        torchvision.transforms.RandomRotation(0.2),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
    ])

    print(data.__len__())

    dl = torch.utils.data.DataLoader(data, batch_size=32, num_workers=2,
                                     shuffle=True)