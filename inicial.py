import cv2
from ultralytics import YOLO
import os

# 1. Configuração de Rede para Câmera IP (Vital para Yoosee)
# Isso diz ao sistema para não esperar um fluxo perfeito e usar TCP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

print("Carregando modelo YOLOv8-Pose...")
modelo = YOLO('yolov8n-pose.pt')

# 2. Seu novo link testado
url_camera = "rtsp://admin:testedacasa01@10.0.10.228:554/onvif1"

print(f"Conectando ao RTSP: {url_camera}")

# CAP_FFMPEG é o motor que o VLC também usa por baixo dos panos
cap = cv2.VideoCapture(url_camera, cv2.CAP_FFMPEG)

# Aumentar o buffer pode ajudar se a rede oscilar
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

if not cap.isOpened():
    print("ERRO: O OpenCV não conseguiu abrir o link RTSP.")
    print("Dica: Feche o VLC antes de rodar, algumas câmeras só aceitam uma conexão por vez!")
    exit()

print("Conectado! Aguardando o fluxo de vídeo...")

while True:
    ret, frame = cap.read()

    if not ret:
        # Em câmeras IP, o primeiro frame pode demorar ou falhar.
        # Em vez de fechar, vamos tentar novamente.
        print("Sinal fraco ou aguardando frame...")
        cv2.waitKey(1000)  # Espera 1 segundo e tenta de novo
        continue

    # Roda a IA
    results = modelo(frame, verbose=False)

    # Plota os resultados (Esqueleto e Caixa)
    annotated_frame = results[0].plot()

    # Mostra o resultado
    cv2.imshow("Monitoramento Real - IA", annotated_frame)

    # Tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Sistema encerrado.")