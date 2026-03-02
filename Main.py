#!/usr/bin/env python3
# Shebang: permite ejecutar este script directamente en Linux/Raspberry Pi (ej. ./script.py)
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from picamera2 import Picamera2
from PIL import Image
from datetime import datetime
import pandas as pd
import os
import time

# =========================================
# CONFIGURACIÓN
# =========================================
MODEL_PATH = "modelo_mobilenetv3_xml_bbox.pth"  # Ruta del modelo entrenado (pesos .pth)
SAVE_INTERVAL = 30         # segundos entre cada grupo de capturas automáticas (cada "ráfaga")
BURST_COUNT = 5            # cuántas fotos seguidas se guardan por cada intervalo
BURST_DELAY = 0.2          # pausa entre cada foto dentro de la ráfaga
OUTPUT_DIR = "Diesel_data" # carpeta donde se guardarán imágenes y el CSV final
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # usa GPU si existe, si no CPU
FRAME_WIDTH, FRAME_HEIGHT = 1440, 1080  # resolución del stream de cámara

# Crear carpeta de salida si no existe (no falla si ya existe)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5 ROIs
# Zonas fijas (x, y, w, h) en donde el sistema "lee" dígitos/elementos de interés
ROIS = [
    (5, 350, 200, 250),
    (250, 350, 200, 250),
    (550, 350, 200, 250),
    (840, 350, 200, 250),
    (1150, 350, 200, 250)
]

# =========================================
# FILTRO USADO EN ENTRENAMIENTO
# =========================================
def preprocess_image(img):
    # Convierte la imagen PIL (RGB) a formato OpenCV BGR para aplicar filtros clásicos de visión
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Convierte a escala de grises para facilitar realce/segmentación
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # CLAHE: ecualización adaptativa del histograma para mejorar contraste (útil con iluminación irregular)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Umbral Otsu: binariza automáticamente separando fondo/figura (imagen blanco/negro)
    _, otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Retorna:
    # 1) imagen filtrada en RGB (para alimentar al modelo, si se espera 3 canales)
    # 2) imagen filtrada en escala de grises binaria (útil para visualizar/depurar)
    return cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB), otsu

# =========================================
# TRANSFORMACIÓN
# =========================================
def transform_for_model(pil_img):
    # Transformaciones para adaptar el ROI a la entrada del modelo:
    # - Resize: fuerza tamaño fijo (128x128)
    # - ToTensor: convierte PIL -> Tensor [C,H,W] con valores 0..1
    # - Normalize: normalización típica (mean/std) estilo ImageNet
    t = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return t(pil_img)

# =========================================
# CARGA DEL MODELO
# =========================================
num_classes = 10  # el modelo clasifica 10 clases (normalmente dígitos 0-9)

# Se crea la arquitectura MobileNetV3 Small sin pesos preentrenados (weights=None)
model = models.mobilenet_v3_small(weights=None)

# Se reemplaza la última capa (clasificador final) para que saque 10 clases
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

# Se cargan los pesos entrenados desde el archivo .pth (mapeando a CPU/GPU según DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# Se mueve el modelo al dispositivo correcto (GPU o CPU) y se coloca en modo evaluación
model = model.to(DEVICE)
model.eval()

# =========================================
# CÁMARA
# =========================================
# Inicializa Picamera2 y configura un preview stream en RGB888 con la resolución deseada
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
))
picam2.start()

# Pequeña espera para estabilizar el stream
cv2.waitKey(1000)

# Ventana principal donde se dibujan ROIs y predicciones
cv2.namedWindow("Predicción dígitos", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Predicción dígitos", FRAME_WIDTH, FRAME_HEIGHT)

# =========================================
# LOOP PRINCIPAL
# =========================================
data_records = []                # lista donde se guardarán registros (timestamp, predicciones, nombre de imagen)
last_save_time = time.time()     # tiempo inicial para controlar guardado cada SAVE_INTERVAL

try:
    while True:
        # Captura un frame completo desde la cámara
        frame = picam2.capture_array()

        # Copia para dibujar encima (rectángulos y texto) sin modificar el frame original
        display = frame.copy()

        # Contenedores para:
        # - ROIs filtrados (para mostrar ventana de depuración)
        # - predicciones por ROI
        filtered_rois = []
        predictions = []

        # === Procesar ROIs ===
        for idx, (x, y, w, h) in enumerate(ROIS):
            # Recorta cada ROI desde el frame completo
            roi = frame[y:y+h, x:x+w]

            # Convierte ROI a PIL (asegurando el orden correcto de canales)
            pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

            # Aplica el mismo preprocesamiento usado durante entrenamiento (CLAHE + Otsu)
            filtered_rgb, filtered_gray = preprocess_image(pil_roi)

            # Guarda versión binaria para mostrarla después en la ventana "Filtros_ROIs"
            filtered_rois.append(filtered_gray)

            # Prepara el ROI filtrado para el modelo:
            # - transforma a tensor
            # - agrega dimensión batch con unsqueeze(0)
            # - envía al DEVICE
            tensor = transform_for_model(Image.fromarray(filtered_rgb)).unsqueeze(0).to(DEVICE)

            # Inferencia sin gradientes (más rápido y eficiente en memoria)
            with torch.no_grad():
                output = model(tensor)                 # salida del modelo (logits)
                pred = torch.argmax(output, 1).item()  # clase con mayor probabilidad (0..9)

            # Guarda la predicción para esta ROI
            predictions.append(pred)

            # Dibuja el rectángulo de la ROI en la imagen de display
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Escribe la predicción sobre la imagen (arriba de cada ROI)
            cv2.putText(display, str(pred), (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Mostrar imagen con predicciones (ventana principal)
        cv2.imshow("Predicción dígitos", display)

        # Mostrar los ROIs filtrados (ventana de depuración/visualización del preprocesamiento)
        if filtered_rois:
            # Concatena horizontalmente los ROIs filtrados (todos redimensionados a 150x150)
            combined = cv2.hconcat([cv2.resize(f, (150, 150)) for f in filtered_rois])
            cv2.imshow("Filtros_ROIs", combined)

        # === Guardado automático cada SAVE_INTERVAL ===
        current_time = time.time()

        # Si ya pasó el intervalo definido, se dispara una ráfaga de guardado
        if current_time - last_save_time >= SAVE_INTERVAL:
            # Reiniciamos el temporizador inmediatamente
            last_save_time = current_time  

            # Convierte la lista de predicciones a string, ej. [1,2,3,4,5] -> "12345"
            pred_str = "".join(map(str, predictions))

            # Timestamp base (año/mes/día_hora/min/seg)
            timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")

            print(f"[CAPTURA] Guardando ráfaga de {BURST_COUNT} imágenes ({pred_str})...")

            # Captura BURST_COUNT imágenes seguidas y las guarda en disco
            for i in range(BURST_COUNT):
                frame_burst = picam2.capture_array()

                # Timestamp con milisegundos (recortado a 3 decimales)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                # Nombre del archivo incluye:
                # - timestamp
                # - predicción concatenada de las ROIs
                # - número de imagen dentro de la ráfaga
                img_name = f"{timestamp}_pred{pred_str}_img{i+1}.png"
                save_path = os.path.join(OUTPUT_DIR, img_name)

                # Guarda la imagen capturada
                cv2.imwrite(save_path, frame_burst)

                # Guarda registro (metadatos) para luego escribir el CSV
                data_records.append({
                    "timestamp": timestamp,
                    "pred_rois": pred_str,
                    "img_name": img_name
                })

                # Pausa entre capturas de la ráfaga
                time.sleep(BURST_DELAY)

            print(f"[OK] Ráfaga guardada correctamente en {OUTPUT_DIR}\n")

        # Salir con ‘q’
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    # Permite terminar con Ctrl+C sin mostrar error feo
    pass
finally:
    # Guardar CSV al final (solo si hubo datos)
    if data_records:
        df = pd.DataFrame(data_records)
        csv_path = os.path.join(OUTPUT_DIR, "Diesel_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"📁 Datos guardados en {csv_path}")

    # Cierre correcto de cámara y ventanas
    picam2.stop()
    cv2.destroyAllWindows()