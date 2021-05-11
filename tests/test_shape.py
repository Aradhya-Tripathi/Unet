import unittest
import torch
from core.unet import Unet


class ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Unet(in_channels=3, n_classes=2)

    def test_out(self):
        test = torch.rand(1, 3, 120, 120)
        self.assertEqual(self.model(test).shape, torch.Size([1, 2, 120, 120]))
        test = torch.rand(1, 3, 161, 161)
        self.assertEqual(self.model(test).shape, torch.Size([1, 2, 161, 161]))
        test = torch.rand(1, 3, 148, 148)
        self.assertEqual(self.model(test).shape, torch.Size([1, 2, 148, 148]))


if __name__ == '__main__':
    unittest.main()
