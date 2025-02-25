from dogsVsCatsCNN import dogsCatsCNN
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from PIL import Image

device = 'cpu'
model = dogsCatsCNN().to(device)
model.load_state_dict(torch.load('DogsVsCatsProject/dogs_catsCNN.pth', map_location=device))
class_names = ['Gato', 'Perro']
model.eval()

transforms = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_images(imgs):
    processed_imgs = []

    for imgs_path in imgs:
        img = Image.open(imgs_path).convert('RGB')
        img = transforms(img).unsqueeze(0)
        processed_imgs.append(img)

    return torch.cat(processed_imgs, dim=0)  # Concatenar imagenes en un solo tensor

def predict(imgs):
    images = load_images(imgs)
    pred = []

    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities > 0.5).float().squeeze()
        
        pred = [class_names[int(idx)] for idx in predicted.tolist()] # Convertir los índices de clases a nombres de clases
    
    return pred

imgs = [
    'DogsVsCatsProject/dog1.jpg', # Mala prediccion
    'DogsVsCatsProject/dog2.jpg', # Buena
    'DogsVsCatsProject/dog3.jpg', # Buena
    'DogsVsCatsProject/dog4.jpg', # Buena
    'DogsVsCatsProject/dog5.jpeg', # Buena
    'DogsVsCatsProject/dog6.jpg', # Buena
    'DogsVsCatsProject/cat1.jpeg' # Mala
]

predictions = predict(imgs)
print(f'Predicciones: {predictions}')

fig, axes = plt.subplots(1, len(imgs), figsize=(10, 5))

# Si solo hay una imagen, axes no es una lista, así que lo convertimos en una lista
if len(imgs) == 1:
    axes = [axes]

for i, (img_path, pred) in enumerate(zip(imgs, predictions)):
    img = Image.open(img_path).convert('RGB')
    ax = axes[i]
    ax.imshow(img)
    ax.set_xticks([])  # Ocultar ticks en X
    ax.set_yticks([])  # Ocultar ticks en Y
    ax.set_title(f'Predicción: {pred}')

plt.show()