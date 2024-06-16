import albumentations as albu
import numpy as np
from PIL import Image
import torch
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda'

def to_tensor(x, **kwargs):
    return torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)

def to_tensor_mask(x, **kwargs):
    return torch.tensor(x, dtype=torch.long)

def get_transforms():
    _transform = [
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        albu.Lambda(image=to_tensor, mask=to_tensor_mask),
    ]
    return albu.Compose(_transform)

def load_image(image_path):
    """
    Load and preprocess image using Albumentations.
    """
    image = np.array(Image.open(image_path).convert('RGB'))
    transform = get_transforms()
    augmented = transform(image=image)
    image = augmented['image']
    return image.unsqueeze(0)  # Add batch dimension

def predict_image(model, image_path):
    """
    Predict the class of image.
    """
    model.eval()  # Set the model to evaluation mode
    image = load_image(image_path).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

import pandas as pd

model_path = 'path_to_best_model'
model = torch.load(model_path)
model.to(DEVICE)

df = pd.read_csv('path_to_test_df')

df['predicted_class'] = df['Ruta'].apply(lambda x: predict_image(model, x))

true_classes = df['Clase'].to_numpy()
predicted_classes = df['predicted_class'].to_numpy()

conf_matrix = confusion_matrix(true_classes, predicted_classes, labels=[0, 1, 2])

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
plt.savefig('path_for_conf_matrix', dpi=300, bbox_inches='tight')
plt.close()

print("Matriu de Confusió:")
print(conf_matrix)


precision = precision_score(true_classes, predicted_classes, average=None, labels=[0, 1, 2])
recall = recall_score(true_classes, predicted_classes, average=None, labels=[0, 1, 2])
f1 = f1_score(true_classes, predicted_classes, average=None, labels=[0, 1, 2])

for cls in [0, 1, 2, 3]:
    print(f"Class {cls} Precision: {precision[cls]:.4f}")
    print(f"Class {cls} Recall: {recall[cls]:.4f}")
    print(f"Class {cls} F1-score: {f1[cls]:.4f}")

precision_macro = precision_score(true_classes, predicted_classes, average='macro')
recall_macro = recall_score(true_classes, predicted_classes, average='macro')
f1_macro = f1_score(true_classes, predicted_classes, average='macro')

print(f"Macro Precision: {precision_macro:.4f}")
print(f"Macro Recall: {recall_macro:.4f}")
print(f"Macro F1-score: {f1_macro:.4f}")
