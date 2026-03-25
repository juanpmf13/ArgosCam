import cv2
import os
import glob

# --- CONFIGURAÇÕES ---
# Onde estão os vídeos que possuem o mapa de profundidade ao lado
pasta_videos = r'C:\ArgosCam\downloads_segundo_dataset'
pasta_destino_raiz = r'C:\ArgosCam\dataset_extraido_limpo'
pular_frames = 1  # 1 extrai tudo.

# Procura por vídeos
extensoes = ['*.mp4', '*.avi', '*.mkv', '*.mpg']
arquivos_video = []
for ext in extensoes:
    arquivos_video.extend(glob.glob(os.path.join(pasta_videos, ext)))

print(f"Encontrados {len(arquivos_video)} vídeos para processar (Com Corte de Profundidade).")

for video_path in arquivos_video:
    nome_video = os.path.basename(video_path).split('.')[0]
    pasta_video_especifico = os.path.join(pasta_destino_raiz, nome_video)

    os.makedirs(pasta_video_especifico, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0

    print(f"Extraindo apenas RGB de: {nome_video}...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        if count % pular_frames == 0:
            # LÓGICA DE CORTE:
            # O frame original tem largura dobrada (Profundidade | RGB)
            height, width, _ = frame.shape
            metade = width // 2

            # Pegamos da metade até o final da largura (Lado Direito - RGB)
            frame_rgb = frame[:, metade:]

            nome_frame = f"{frame_id:05d}.jpg"
            cv2.imwrite(os.path.join(pasta_video_especifico, nome_frame), frame_rgb)
            frame_id += 1

        count += 1

    cap.release()

print("\n✅ Concluído! Apenas a parte visual (RGB) foi extraída em: " + pasta_destino_raiz)