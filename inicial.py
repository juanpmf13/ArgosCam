import cv2
from ultralytics import YOLO
import os

# 1. Caminhos Garantidos
caminho_modelo = r"C:\ArgosCam\yolov8n-pose.pt"
modelo = YOLO(caminho_modelo)
print("IA PRONTA!")

# 2. Tentar abrir a Webcam
# Se 0 não funcionar, tente 1 ou -1
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW ajuda no Windows

if not cap.isOpened():
    print("Erro: Webcam não encontrada ou ocupada por outro app.")
else:
    print("Webcam detectada! Abrindo janela...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem.")
            break

        # Roda a detecção de pose
        results = modelo(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Mostra o resultado
        cv2.imshow("Teste ArgosCam", annotated_frame)

        # 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()