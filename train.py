import torch
from torch.nn.functional import log_softmax
from torchvision import transforms
from tqdm import tqdm


class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim,
                 criterion: torch.nn.CrossEntropyLoss):
        self.model = model
        self.optim = optimizer
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.model = model.to(self.device)

    def train(self, train_loader) -> tuple:
        """
        Args:
            model
            optim
            dl

        Return:
            epoch_loss
            epoch_acc

        """
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0

        for xid, (image, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            self.optim.zero_grad()
            image, label = image.to(self.device), label.to(self.device)
            out = self.model(image)
            loss = self.criterion(out, label)

            loss.backward()

            self.optim.step()

            epoch_loss += loss.item()
            epoch_acc += self.accuracy(label, out).item()

        return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


if '__main__' == __name__:
    from core.unet import Unet
    from core.data import data

    model = Unet(in_channels=3, n_classes=2)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, optim, criterion)

    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data = data(path_to_data='../train', transforms=trans)
    train_split, test_split = int(len(data) * 0.8), int(len(data) * 0.2)
    train_loader, test_loader = data.get_data(32, [train_split, test_split])

    epochs = 7

    for i in range(epochs):
        loss, acc = trainer.train(train_loader)
        print(f'EPOCH : {i + 1} ] LOSS : {loss} ] ACC: {acc * 100}')
