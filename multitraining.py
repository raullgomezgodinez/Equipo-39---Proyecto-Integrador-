#!/usr/bin/env python3
"""
multitraining.py

Este script entrena y compara múltiples arquitecturas de visión por computadora para un problema
de clasificación basado en objetos anotados con bounding boxes (formato tipo VOC: imagen + XML).

Idea general:
1) Se recorren imágenes del dataset y se buscan sus XML correspondientes.
2) Por cada objeto anotado en el XML se recorta (crop) el bounding box y ese recorte se trata como
   una muestra para clasificación (imagen recortada -> clase del objeto).
3) Se divide el conjunto total de recortes en entrenamiento y validación.
4) Se entrena cada modelo definido en MODEL_LIST y se guarda el mejor checkpoint por accuracy.
5) Se imprime un resumen final y se identifica el mejor modelo.

Notas importantes:
- MobileNetV3 Small y ResNet18 se inicializan con pesos preentrenados de ImageNet.
- EfficientNet-B0 se define con weights=None (entrenamiento desde cero). Esto puede afectar la
  comparación, ya que no parte del mismo “punto de inicio” que los modelos preentrenados.
"""

import os
from glob import glob
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models


# =============================================================================
# CONFIGURACIÓN (manteniendo el mismo orden y parámetros)
# =============================================================================

# Directorio del dataset (ruta local). Debe contener imágenes y sus XML con el mismo nombre base.
DATASET_DIR = r"C:\Users\gabri\Desktop\dataset_tulum"

# Lista de modelos a comparar
MODEL_LIST = ["mobilenet_v3_small", "resnet18", "efficientnet_b0"]

# Hiperparámetros
BATCH_SIZE = 16
EPOCHS = 2
LR = 1e-4
IMAGE_SIZE = 224

# Selección de dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# TRANSFORMS
# =============================================================================
"""
Las transformaciones se usan para estandarizar tamaño y normalización de las imágenes, y también
para aplicar data augmentation en entrenamiento.

- Train:
  - Resize a 224x224
  - Rotación aleatoria ligera (10°) para robustez
  - Normalización con media y desviación estándar de ImageNet (compatible con modelos preentrenados)

- Val:
  - Resize a 224x224
  - Sin augmentación (para medir desempeño de forma consistente)
"""

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =============================================================================
# DATASET (VOCDataset) - mismo comportamiento, pero legible
# =============================================================================
class VOCDataset(Dataset):
    """
    Dataset basado en anotaciones VOC (XML):

    Cada imagen puede contener múltiples objetos. En lugar de entrenar un detector,
    este enfoque convierte el problema en clasificación:
    - Se recorta cada objeto usando su bounding box.
    - Ese recorte se etiqueta con el 'name' del objeto en el XML.

    Así, una sola imagen puede generar varias muestras (una por objeto).
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Buscar imágenes
        image_paths = (
            glob(os.path.join(root_dir, "*.jpg")) +
            glob(os.path.join(root_dir, "*.png")) +
            glob(os.path.join(root_dir, "*.JPG"))
        )
        print("Imágenes detectadas:", len(image_paths))

        # Generar samples a partir de objetos anotados
        for img_path in image_paths:
            xml_path = os.path.splitext(img_path)[0] + ".xml"
            if not os.path.exists(xml_path):
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = root.findall("object")
            if not objects:
                continue

            for obj in objects:
                label = obj.find("name").text

                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                # Guardamos: ruta + etiqueta + coordenadas bbox
                self.samples.append((img_path, label, xmin, ymin, xmax, ymax))

        # Construcción de catálogo de clases
        self.classes = sorted(list(set([s[1] for s in self.samples])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        print("Samples cargados:", len(self.samples))
        print("Clases:", self.classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, xmin, ymin, xmax, ymax = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        # Recorte del objeto usando bbox
        image = image.crop((xmin, ymin, xmax, ymax))

        label_idx = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label_idx


# =============================================================================
# CARGA DEL DATASET + SPLIT (train/val) - mismo flujo
# =============================================================================
print("\nCargando dataset...")
full_dataset = VOCDataset(DATASET_DIR, transform=None)
num_classes = len(full_dataset.classes)

"""
En este punto, full_dataset contiene recortes (samples) que representan objetos.
Ahora se divide en entrenamiento y validación.

Importante:
- El split se hace sobre los índices de samples, no sobre archivos de imagen.
- Si una misma imagen tiene varios objetos, algunos recortes podrían caer en train
  y otros en val. Si esto es un problema para tu evaluación, habría que particionar
  por imagen base (no por objeto). Pero aquí se mantiene exactamente la lógica original.
"""
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

# Asignación de transforms a train y val (misma intención que el código original)
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Clases:", full_dataset.classes)
print("Train:", len(train_dataset))
print("Val:", len(val_dataset))


# =============================================================================
# FACTORY DE MODELOS (create_model) - misma lógica, con comentario explicativo
# =============================================================================
def create_model(name, num_classes):
    """
    Crea un modelo según el nombre y reemplaza su última capa para producir num_classes.

    - mobilenet_v3_small: se ajusta model.classifier[3]
    - resnet18: se ajusta model.fc
    - efficientnet_b0: se ajusta model.classifier[1]

    Nota metodológica:
    - MobileNetV3 y ResNet18 usan pesos preentrenados (transfer learning).
    - EfficientNet-B0 se crea sin pesos (weights=None).
      Esto puede reducir su performance al inicio y requiere más épocas para competir.
    """
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, num_classes
        )

    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    else:
        raise ValueError("Modelo no soportado")

    return model


# =============================================================================
# ENTRENAMIENTO/VALIDACIÓN - mismo orden y criterio (accuracy)
# =============================================================================
criterion = nn.CrossEntropyLoss()
results = {}

"""
Entrenamos cada modelo en MODEL_LIST con el mismo dataset y misma cantidad de épocas.
La métrica usada es accuracy sobre el conjunto de validación.

Se guarda un checkpoint cuando mejora el accuracy.
Esto permite comparar modelos por su mejor desempeño observado durante el entrenamiento.
"""

for model_name in MODEL_LIST:
    print("\nEntrenando:", model_name)

    model = create_model(model_name, num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # -----------------------------
        # Fase de entrenamiento
        # -----------------------------
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # -----------------------------
        # Fase de validación
        # -----------------------------
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                preds = model(images).argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100.0 * correct / max(1, total)
        print("Accuracy:", acc)

        # Guardado de mejor checkpoint por modelo
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"modelo_{model_name}.pth")

    # Guardar el mejor accuracy alcanzado por este modelo
    results[model_name] = best_acc


# =============================================================================
# RESUMEN FINAL DE RESULTADOS - mismo flujo
# =============================================================================
print("\nRESULTADOS FINALES")
for name, acc in results.items():
    print(name, "=", acc)

best = max(results, key=results.get)
print("MEJOR MODELO:", best)

"""
Cómo “compartir resultados” en un reporte o exposición:

- Reporta el best accuracy por modelo (lo que ya imprime el script).
- Menciona qué modelos parten de pesos preentrenados y cuál no.
- Indica hiperparámetros: epochs, batch_size, LR, IMAGE_SIZE.
- Describe el enfoque: recorte por bounding box (VOC) -> clasificación por objeto.
- Aclara cómo se validó: accuracy en conjunto de validación (20% split).

Si quieres evidencias más completas (sin cambiar lógica), lo típico es:
- Guardar results en un .json o .csv
- Guardar loss promedio por epoch
Pero aquí dejamos el script con la misma salida (prints y checkpoints).
"""
