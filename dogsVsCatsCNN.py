import torch # type: ignore
import torch.nn.functional as F # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torchvision import transforms, datasets # type: ignore

transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10, expand=False),
    transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.ColorJitter(0.3, 0.3),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_loaders(batch_size=64, use_workers=True): # Se bajo de 128 a 64 (pendiente de prueba)
    
    num_workers = 2 if use_workers else 0

    train_dataset =  datasets.ImageFolder(root='Users/Admin/Documents/pytorch_projects/dogs_vs_cats/train', transform=transform)
    test_dataset =  datasets.ImageFolder(root='Users/Admin/Documents/pytorch_projects/dogs_vs_cats/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    for x, y in test_loader:
        print(f'x shape: {x.shape}')
        print(f'y shape: {y.shape}{y.dtype}')
        break

    return train_loader, test_loader

class dogsCatsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*4*4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        y = y.float().unsqueeze(1)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f'Loss: {loss:>7f}, progress: [{current:>5d}/{size:>5d}]')

    #scheduler.step(loss)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    batch_size = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            y = y.float().unsqueeze(1)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += ((pred > 0.5).float() == y).sum().item()

    test_loss /= batch_size
    correct /= size
    print(f'Test-loss: {test_loss:>7f}, accuracy: {(100 * correct):>0.1f}%')

if __name__ == '__main__':
    device = 'cpu'
    epochs = 40

    train_loader, test_loader = get_loaders()

    model = dogsCatsCNN().to(device)
    print(model)

    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                    optimizer, 
                                                    mode='min', 
                                                    factor=0.2,
                                                    patience=5,
                                                    threshold_mode='rel'
                                                )

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train(train_loader, model, loss, optimizer)
        test(test_loader, model, loss)

        last_lr = scheduler.get_last_lr()
        print(f'Last lr: {last_lr}')
        print("-------------------------------")

    print("Finished")

    torch.save(model.state_dict(), 'DogsVsCatsProject/dogs_catsCNN_sch.pth')
    print("Weights saved")