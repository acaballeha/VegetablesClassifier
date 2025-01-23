

from PIL import Image
import torch
from torchvision import models, transforms
import argparse

# Cargar el modelo preentrenado
from torchvision.models import ResNet50_Weights

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_classes = 15  # Adjust this to the number of classes in your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
pesos = torch.load('./best_model_weights.pth', weights_only=True) # Carga de los pesos del modelo entrenado para clasificar vegetales
model.load_state_dict(pesos)

## Preprocesamiento de la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
   
])

categories = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd',
              'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 
              'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 
              'Potato', 'Pumpkin', 'Radish', 'Tomato']





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Classify vegetables')
    parser.add_argument('image', type=str, help='Path to the image file')
    args = parser.parse_args()
    
    img = Image.open(args.image)
    img_t = transform(img)
    model.eval()
  
    batch_t = torch.unsqueeze(img_t, 0)
    out = model(batch_t)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    top3 = torch.topk(percentage, 3)

    # print(f'Confidence: {percentage[index[0]].item():.2f}%, Class: {categories[index[0].item()]}')
    # print('The image is a:', categories[index[0].item()])

    print(f'The image {args.image} is:')
    for i in range(top3.indices.size(0)):
        print(f'\t{categories[top3.indices[i].item()]}: {top3.values[i].item():.2f}%')
    
    
