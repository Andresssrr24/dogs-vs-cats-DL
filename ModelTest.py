from dogsVsCatsCNN import dogsCatsCNN, get_loaders
import matplotlib.pyplot as plt # type:ignore
import torch # type:ignore

class_names = ['Gato', 'Perro']

_, test_dataloader = get_loaders(use_workers=False)
dataiter = iter(test_dataloader)
images, labels = next(dataiter)

device = "cpu"
images = images.to(device)

# Carga del modelo
model = dogsCatsCNN().to(device)
model.load_state_dict(torch.load('DogsVsCatsProject/dogs_catsCNN.pth', map_location=device))
model.eval()

def predictions(model, images, labels):
    # Predicciones
    with torch.no_grad():
        outputs = model(images)
        predicted = (outputs > 0.5).float().squeeze()  # En clasificacion binaria solo se recibe un tensor

        print(f"Predicciones: {predicted}")
    
    # Desnormalizacion de imagenes
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_denorm = images * std + mean
    images_denorm = images_denorm.clamp(0, 1)

    images_denorm = images_denorm.permute(0, 2, 3, 1).cpu().numpy()  # Convertir a formato (h, w, c)

    # Representacion grafica
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.5)
    axes = axes.ravel()

    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            ax.axis('off')
            continue

        image = images_denorm[i]
        true_label = class_names[labels[i].item()]
        pred_label = class_names[int(predicted[i].item())]

        ax.imshow(image)
        ax.set_xticks([])  # Quitar ticks en X
        ax.set_yticks([])  # Quitar ticks en Y
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10)

    plt.show()

predictions(model, images, labels)
