#!/usr/bin/env python3
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
MODEL_PATH = "modelo_mobilenetv3_xml_bbox.pth"
SAVE_INTERVAL = 30         # segundos entre cada grupo de capturas
BURST_COUNT = 5            # cuántas fotos seguidas por intervalo
BURST_DELAY = 0.2          # segundos entre cada foto
OUTPUT_DIR = "Diesel_data" # carpeta de salida
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_WIDTH, FRAME_HEIGHT = 1440, 1080

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5 ROIs
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
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    _, otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB), otsu

# =========================================
# TRANSFORMACIÓN
# =========================================
def transform_for_model(pil_img):
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
num_classes = 10
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =========================================
# CÁMARA
# =========================================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
))
picam2.start()
cv2.waitKey(1000)

cv2.namedWindow("Predicción dígitos", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Predicción dígitos", FRAME_WIDTH, FRAME_HEIGHT)

# =========================================
# LOOP PRINCIPAL
# =========================================
data_records = []
last_save_time = time.time()  # tiempo de referencia inicial

try:
    while True:
        frame = picam2.capture_array()
        display = frame.copy()
        filtered_rois = []
        predictions = []

        # === Procesar ROIs ===
        for idx, (x, y, w, h) in enumerate(ROIS):
            roi = frame[y:y+h, x:x+w]
            pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            filtered_rgb, filtered_gray = preprocess_image(pil_roi)
            filtered_rois.append(filtered_gray)

            tensor = transform_for_model(Image.fromarray(filtered_rgb)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(tensor)
                pred = torch.argmax(output, 1).item()
            predictions.append(pred)

            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, str(pred), (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Mostrar imagen con predicciones
        cv2.imshow("Predicción dígitos", display)

        # Mostrar los ROIs filtrados
        if filtered_rois:
            combined = cv2.hconcat([cv2.resize(f, (150, 150)) for f in filtered_rois])
            cv2.imshow("Filtros_ROIs", combined)

        # === Guardado automático cada SAVE_INTERVAL ===
        current_time = time.time()
        if current_time - last_save_time >= SAVE_INTERVAL:
            # Reiniciamos el temporizador inmediatamente
            last_save_time = current_time  

            pred_str = "".join(map(str, predictions))
            timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")

            print(f"[CAPTURA] Guardando ráfaga de {BURST_COUNT} imágenes ({pred_str})...")

            for i in range(BURST_COUNT):
                frame_burst = picam2.capture_array()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                img_name = f"{timestamp}_pred{pred_str}_img{i+1}.png"
                save_path = os.path.join(OUTPUT_DIR, img_name)

                cv2.imwrite(save_path, frame_burst)
                data_records.append({
                    "timestamp": timestamp,
                    "pred_rois": pred_str,
                    "img_name": img_name
                })

                time.sleep(BURST_DELAY)

            print(f"[OK] Ráfaga guardada correctamente en {OUTPUT_DIR}\n")

        # Salir con ‘q’
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    # Guardar CSV
    if data_records:
        df = pd.DataFrame(data_records)
        csv_path = os.path.join(OUTPUT_DIR, "Diesel_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"📁 Datos guardados en {csv_path}")

    picam2.stop()
    cv2.destroyAllWindows()