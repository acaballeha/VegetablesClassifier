

from PIL import Image
import torch
from torchvision import models, transforms
import argparse

# Cargar el modelo preentrenado
from torchvision.models import ResNet50_Weights
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk
import os


model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_classes = 15  # Adjust this to the number of classes in your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
pesos = torch.load('./best_model_weights.pth', weights_only=True) # Carga de los pesos del modelo entrenado para clasificar vegetales
model.load_state_dict(pesos)

## Preprocesamiento de la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
   
])

categories = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd',
              'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 
              'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 
              'Potato', 'Pumpkin', 'Radish', 'Tomato']





if __name__ == '__main__':
    print (tk.TkVersion)
    def classify_image(image_path):
        img = Image.open(image_path)
        img_t = transform(img)
        model.eval()
        batch_t = torch.unsqueeze(img_t, 0)
        out = model(batch_t)
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        top3 = torch.topk(percentage, 3)
        
        basename = os.path.basename(image_path)
        result = f'The image {basename} is:\n'
        for i in range(top3.indices.size(0)):
            result += f'\t{categories[top3.indices[i].item()]}: {top3.values[i].item():.2f}%\n'
        return result

    def open_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            result = classify_image(file_path)
            result_label.config(text=result)
            img = Image.open(file_path)
            img.thumbnail((200, 200))
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img

    root = tk.Tk()
    root.title("Vegetable Classifier")
    root.geometry("600x200")

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    image_label = ttk.Label(frame)
    image_label.grid(row=0, column=0, rowspan=2, padx=10, pady=10)

    result_label = ttk.Label(frame, text="Use the button to open a file")
    result_label.grid(row=0, column=1, padx=10, pady=10)

    open_button = ttk.Button(frame, text="Open Image", command=open_file)
    open_button.grid(row=1, column=1, padx=10, pady=10)

    root.mainloop()
