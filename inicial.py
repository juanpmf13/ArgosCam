import subprocess
import numpy as np
import cv2
from ultralytics import YOLO
import os

# 1. Configurações de Conexão (O que funcionou agora)
width = 1920
height = 1080
url = "rtsp://admin:testedacasa01@10.0.10.228:554/onvif1"
ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg.exe")

command = [
    ffmpeg_path,
    '-loglevel', 'quiet',
    '-probesize', '10M',
    '-i', url,
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo',
    '-sn', '-an',
    '-'
]

# 2. Inicialização
print("Carregando IA YOLOv8...")
modelo = YOLO(r"C:\ArgosCam\yolov8n-pose.pt")

print("Iniciando fluxo de vídeo...")
process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10 ** 8)

while True:
    # Captura o frame bruto
    bytes_to_read = width * height * 3
    raw_frame = process.stdout.read(bytes_to_read)

    if not raw_frame or len(raw_frame) != bytes_to_read:
        continue

    # Transforma em imagem numpy
    frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

    # Redimensiona para 640px (O YOLO trabalha melhor e mais rápido nessa resolução)
    frame_ia = cv2.resize(frame, (640, 640))

    # Executa a Detecção de Pose
    results = modelo(frame_ia, verbose=False)
    annotated_frame = results[0].plot()

    # Lógica de Alerta de Queda
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            # Se a largura for muito maior que a altura, indica queda
            if w > (h * 1.2):
                cv2.putText(annotated_frame, "ALERTA: QUEDA!", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Exibe o resultado final
    cv2.imshow("ArgosCam - Protecao Ativa", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process.terminate()
cv2.destroyAllWindows()