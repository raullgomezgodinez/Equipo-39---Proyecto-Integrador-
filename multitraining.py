#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import os
from PIL import Image
import xml.etree.ElementTree as ET
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# =========================================
# CONFIG
# =========================================

DATASET_DIR = r"C:\Users\gabri\Desktop\dataset_tulum"

MODEL_LIST = [

    "mobilenet_v3_small",

    "resnet18",

    "efficientnet_b0"
]

BATCH_SIZE = 16
EPOCHS = 2
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224


# =========================================
# TRANSFORMS
# =========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# =========================================
# DATASET
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
# LOAD DATASET
# =========================================

print("\nCargando dataset...")

full_dataset = VOCDataset(DATASET_DIR)

num_classes = len(full_dataset.classes)

# 🔹 Crear índices
indices = list(range(len(full_dataset)))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42
)

# 🔹 Subsets
train_dataset = Subset(full_dataset, train_idx)
val_dataset   = Subset(full_dataset, val_idx)

# 🔹 Aplicar transforms
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform   = val_transform

# 🔹 DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

print("Clases:", full_dataset.classes)
print("Train:", len(train_dataset))
print("Val:", len(val_dataset))


# =========================================
# CREATE MODEL
# =========================================

def create_model(name,num_classes):

    if name=="mobilenet_v3_small":

        model=models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        model.classifier[3]=nn.Linear(
            model.classifier[3].in_features,
            num_classes
        )


    elif name=="resnet18":

        model=models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        model.fc=nn.Linear(
            model.fc.in_features,
            num_classes
        )


    elif name=="efficientnet_b0":

        model = models.efficientnet_b0(weights=None)
        

        model.classifier[1]=nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )


    return model


# =========================================
# TRAIN
# =========================================

criterion = nn.CrossEntropyLoss()

results = {}


for MODEL_NAME in MODEL_LIST:


    print("\n===================================")
    print("Entrenando:",MODEL_NAME)
    print("===================================\n")


    model=create_model(
        MODEL_NAME,
        num_classes
    )

    model=model.to(DEVICE)

    optimizer=optim.Adam(
        model.parameters(),
        lr=LR
    )


    best_acc=0


    for epoch in range(EPOCHS):


        print(f"\nEpoch {epoch+1}/{EPOCHS}")


        model.train()

        running_loss=0


        progress_bar=tqdm(
            train_loader,
            desc="Entrenando",
            leave=False
        )


        for images,labels in progress_bar:


            images,labels=images.to(DEVICE),labels.to(DEVICE)


            optimizer.zero_grad()

            outputs=model(images)

            loss=criterion(outputs,labels)

            loss.backward()

            optimizer.step()


            running_loss+=loss.item()


            progress_bar.set_postfix(
                loss=loss.item()
            )


        avg_loss=running_loss/len(train_loader)


        # VALIDACION

        model.eval()

        correct=0
        total=0


        with torch.no_grad():

            for images,labels in val_loader:

                images,labels=images.to(DEVICE),labels.to(DEVICE)

                preds=model(images).argmax(1)

                total+=labels.size(0)

                correct+=(preds==labels).sum().item()


        acc=100*correct/total


        print(
            f"Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%"
        )


        if acc>best_acc:

            best_acc=acc

            torch.save(
                model.state_dict(),
                f"modelo_{MODEL_NAME}.pth"
            )


    results[MODEL_NAME]=best_acc


# =========================================
# RESULTADOS
# =========================================

print("\n===================================")
print("RESULTADOS FINALES")
print("===================================\n")


for name,acc in results.items():

    print(name,"=",round(acc,2),"%")


best=max(results,key=results.get)


print("\nMEJOR MODELO:",best)