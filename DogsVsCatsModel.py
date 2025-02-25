import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Transformaciones a las imagenes
transforms = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10, expand=False), 
    #transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    #transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def get_dataloaders(batch_size=128, use_workers=True):
    num_workers = 2 if use_workers else 0

    # Cargar dataset
    train_dataset = datasets.ImageFolder(root="DogsVsCatsProject/dogs_vs_cats/train", transform=transforms)
    test_dataset = datasets.ImageFolder(root="DogsVsCatsProject/dogs_vs_cats/test", transform=transforms)

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # workers -> Procesos en 2do plano que cargan los datos en los batches
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    for X, y in test_dataloader:
        print(f'Shape of X: {X.shape}')
        print(f'Shape of y: {y.shape} {y.dtype}')
        break
    
    return train_dataloader, test_dataloader

# Modelo
class BinaryAnimalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Linear(3*64*64, 8192),
            nn.ReLU(),
            nn.BatchNorm1d(8192),
            nn.Dropout(0.5),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    def forward(self, X):
        X = self.flatten(X)
        logits = self.layer_stack(X)
        return logits 

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        y = y.float().unsqueeze(1)  # Cambiar el tamaño del tensor para procesar imgs RGB [3*128*128, 1]
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if batch % 10 == 0:
        loss, current = loss.item(), (batch + 1) * len(X)   
        print(f'Loss: {loss:>7f}, Current: {current:>5d}/Size: {size:>5d}') 

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.float().unsqueeze(1)  # Cambiar el tamaño del tensor para procesar imgs RGB [3*128*128, 1]

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += ((pred > 0.5).float() == y).sum().item()  # Procesar las predicciones en base a la clasificacion binaria (una clase o la otra)
    test_loss /= num_batches
    correct /= size
    print(f'Test acc: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n')

    # ----------------------------------------------------------------
if __name__ == "__main__":
    train_dataloader, test_dataloader = get_dataloaders()

    device = "cpu"
    model = BinaryAnimalNN().to(device)
    #model.load_state_dict(torch.load('DogsVsCatsProject/binary_clSecond.pth'))  # Cargar weights pre entrenaddos para usarlos como punto de partida para otro entrenamiento
    print(model)

    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Entrenamiento
    epochs = 180
    print(f"Training for {epochs} epochs"   )
    for t in range(epochs): 
        print(f'Epoch: {t+1}\n-------------------------------')
        train(train_dataloader, model, loss, optimizer)
        test(test_dataloader, model, loss)
    print('Training finished')

    # ----------------------------------------------------------------
    # Guardar modelo
    torch.save(model.state_dict(), 'DogsVsCatsProject/binary_clBigNN-180eV2.pth')
    print(f'Model saved')
        