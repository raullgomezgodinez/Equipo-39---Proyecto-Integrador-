from picamera2 import Picamera2
import cv2
import time

# --- Configuración ---
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
ROIS = [
    (50, 50, 100, 80),   # ROI 1 (x, y, w, h)
    (200, 50, 100, 80),  # ROI 2
    (350, 50, 100, 80)   # ROI 3
]

# --- Inicializar cámara ---
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"format": 'RGB888', "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
)
picam2.start()
time.sleep(1)  # esperar a que el stream se estabilice

# Crear ventana OpenCV
cv2.namedWindow("Cámara OpenCV", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cámara OpenCV", FRAME_WIDTH, FRAME_HEIGHT)

print("📷 Cámara abierta con ROIs. Presiona 'q' para cerrar.")

try:
    while True:
        frame = picam2.capture_array()
        if frame is None or frame.size == 0:
            continue  # esperar frame válido

        # Convertir a BGR para OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Dibujar ROIs
        for x, y, w, h in ROIS:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar ventana
        cv2.imshow("Cámara OpenCV", frame_bgr)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("🛑 Cámara detenida")