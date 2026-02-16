#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xml.etree.ElementTree as ET
from glob import glob
from sklearn.model_selection import train_test_split

# =========================================
# CONFIGURACIÓN
# =========================================
DATASET_DIR = r"C:\Users\gabri\Desktop\dataset_tulum"
MODEL_PATH = "modelo_mobilenetv3_filtrado_dataset_final.pth"
BATCH_SIZE = 32
EPOCHS = 6
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================
# TRANSFORMACIONES
# =========================================
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================================
# DATASET PERSONALIZADO (XML + BBOX CROP)
# =========================================
class VOCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        image_paths = glob(os.path.join(root_dir, "*.jpg")) + \
                      glob(os.path.join(root_dir, "*.png")) + \
                      glob(os.path.join(root_dir, "*.JPG"))

        print(f"🔍 Imágenes detectadas: {len(image_paths)}")

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

                self.samples.append((img_path, label, xmin, ymin, xmax, ymax))

        self.classes = sorted(list(set([s[1] for s in self.samples])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        print(f"✅ Samples cargados: {len(self.samples)}")
        print(f"🏷️ Clases: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, xmin, ymin, xmax, ymax = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = image.crop((xmin, ymin, xmax, ymax))

        label = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================================
# CARGA DATASET
# =========================================
full_dataset = VOCDataset(DATASET_DIR)
num_classes = len(full_dataset.classes)

indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = Subset(
    VOCDataset(DATASET_DIR, transform=train_transform),
    train_idx
)

val_dataset = Subset(
    VOCDataset(DATASET_DIR, transform=val_transform),
    val_idx
)



train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📂 Total objetos detectados: {len(full_dataset)}")
print(f"📈 Train: {len(train_idx)} | Val: {len(val_idx)}")
print(f"🧠 Clases: {full_dataset.classes}")

# =========================================
# MODELO
# =========================================
model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================================
# ENTRENAMIENTO
# =========================================
best_acc = 0.0

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # VALIDACIÓN
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\n🏆 Mejor accuracy: {best_acc:.2f}%")
print(f"💾 Modelo guardado en {MODEL_PATH}")

# =========================================
# MATRIZ DE CONFUSIÓN
# =========================================
y_true, y_pred = [], []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=full_dataset.classes)
disp.plot(cmap="Blues", xticks_rotation="vertical")
plt.show()