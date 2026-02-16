import os
from glob import glob
import xml.etree.ElementTree as ET

DATASET_DIR = r"C:\Users\gabri\Desktop\dataset_tulum"

print("📂 Ruta:", DATASET_DIR)
print("Existe:", os.path.exists(DATASET_DIR))
print("-" * 50)

# Buscar imágenes
image_paths = glob(os.path.join(DATASET_DIR, "*.jpg")) + \
              glob(os.path.join(DATASET_DIR, "*.png")) + \
              glob(os.path.join(DATASET_DIR, "*.JPG"))

print(f"🖼️ Imágenes encontradas: {len(image_paths)}")

if len(image_paths) == 0:
    print("❌ No se encontraron imágenes.")
    exit()

samples = 0
missing_xml = 0
empty_xml = 0

for img_path in image_paths:
    xml_path = os.path.splitext(img_path)[0] + ".xml"

    if not os.path.exists(xml_path):
        missing_xml += 1
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = root.findall("object")

    if not objects:
        empty_xml += 1
        continue

    samples += len(objects)

print("-" * 50)
print(f"📊 Total objetos válidos: {samples}")
print(f"⚠️ Imágenes sin XML: {missing_xml}")
print(f"⚠️ XML sin objetos: {empty_xml}")
print("-" * 50)