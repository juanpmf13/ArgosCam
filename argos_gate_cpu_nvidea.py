import numpy as np
import cv2
import os
import time
from ultralytics import YOLO
from collections import deque
from argos_api import ArgosApiClient  # Importa o módulo isolado que criamos acima

# --- 1. CONFIGURAÇÕES ---
FONTE_ARQUIVO_LOCAL = r"C:\path\to\your_video.mp4"
LINK_CELULAR = "Link your mobile app "

PATH_MODELO = r"C:\path\to\your_model.pt"
PASTA_ALERTAS = r"D:\ArgosCam\alertas"
PASTA_LOGS = r"D:\ArgosCam\logs"
COOLDOWN_SECONDS = 15

# Inicialização e Login automático na API protegida
api_client = ArgosApiClient()
api_client.autenticar()

# Criar pastas locais de persistência
for p in [PASTA_ALERTAS, PASTA_LOGS]:
    if not os.path.exists(p):
        os.makedirs(p)

# Variáveis de Controle de IDs e Estabilização
buffer_frames = deque(maxlen=150)
historicos_por_id = {}
ultimo_salvamento = 0
posicoes_anteriores = {}  # {id: (centro_x, centro_y, timestamp)}
ARQUIVO_LOG = os.path.join(PASTA_LOGS, f"log_{time.strftime('%Y%m%d')}.txt")


def registrar_log(mensagem):
    timestamp = time.strftime("%H:%M:%S")
    with open(ARQUIVO_LOG, "a", encoding='utf-8') as f:
        f.write(f"[{timestamp}] {mensagem}\n")


# --- 2. INICIALIZAÇÃO DA CÂMERA E MODELO ---
print("🧠 Carregando Argos Gate com Estabilização de ID...")
modelo = YOLO(PATH_MODELO)

if FONTE_ARQUIVO_LOCAL.strip() != "" and os.path.exists(FONTE_ARQUIVO_LOCAL):
    cap = cv2.VideoCapture(FONTE_ARQUIVO_LOCAL)
    modo_streaming = False
else:
    cap = cv2.VideoCapture(LINK_CELULAR)
    modo_streaming = True

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS_VIDEO = cap.get(cv2.CAP_PROP_FPS) or 30

# --- 3. LOOP PRINCIPAL DE INFERÊNCIA ---
try:
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if modo_streaming:
                time.sleep(2)
                cap = cv2.VideoCapture(LINK_CELULAR)
                continue
            else:
                break

        tempo_atual = time.time()

        # Cálculo de FPS Real
        fps_real = 1 / (tempo_atual - prev_time) if (tempo_atual - prev_time) > 0 else 0
        prev_time = tempo_atual

        # Frame de visualização
        altura_viz = 450
        proporcao = altura_viz / HEIGHT
        display_frame = cv2.resize(frame, (int(WIDTH * proporcao), altura_viz))

        # IA - Configurações do rastreador
        results = modelo.track(
            frame,
            imgsz=640,
            persist=True,
            conf=0.40,
            iou=0.4,
            tracker="botsort.yaml",
            augment=True
        )

        ids_vivos_neste_frame = []

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box, obj_id, cls in zip(boxes, ids, clss):
                classe_nome = results[0].names[cls]
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # --- LÓGICA DE ESTABILIZAÇÃO DE ID ---
                id_estabilizado = obj_id
                for id_antigo, dados in list(posicoes_anteriores.items()):
                    velho_x, velho_y, velho_ts = dados
                    distancia = np.sqrt((cx - velho_x) ** 2 + (cy - velho_y) ** 2)

                    if distancia < 100 and (tempo_atual - velho_ts) < 3.0:
                        id_estabilizado = id_antigo
                        break

                posicoes_anteriores[id_estabilizado] = (cx, cy, tempo_atual)
                ids_vivos_neste_frame.append(id_estabilizado)

                if id_estabilizado not in historicos_por_id:
                    historicos_por_id[id_estabilizado] = deque(maxlen=60)

                historicos_por_id[id_estabilizado].append(classe_nome)

                # Contagem de frames em estado crítico
                alert_count = historicos_por_id[id_estabilizado].count('caindo') + \
                              historicos_por_id[id_estabilizado].count('caido')

                # Renderização das Bounding Boxes
                vx1, vy1, vx2, vy2 = map(int, box * proporcao)
                cor = (0, 0, 255) if alert_count > 10 else (0, 255, 0)
                cv2.rectangle(display_frame, (vx1, vy1), (vx2, vy2), cor, 2)
                cv2.putText(display_frame, f"PESSOA {id_estabilizado}: {classe_nome}", (vx1, vy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

                # Disparo Condicional do Alerta
                if alert_count >= 25 and (tempo_atual - ultimo_salvamento) > COOLDOWN_SECONDS:
                    ts = time.strftime("%H%M%S")

                    # 1. OpenCV grava usando 'mp4v'
                    nome_arq_bruto = os.path.join(PASTA_ALERTAS, f"RAW_QUEDA_ID{id_estabilizado}_{ts}.mp4")
                    print(f"🚨 [ALERTA] QUEDA DETECTADA! Gravando buffer...")

                    out = cv2.VideoWriter(nome_arq_bruto, cv2.VideoWriter_fourcc(*'mp4v'), FPS_VIDEO, (WIDTH, HEIGHT))
                    for f_buf in buffer_frames:
                        out.write(f_buf)
                    out.release()

                    ultimo_salvamento = tempo_atual

                    # 2. COMPRESSÃO COM FFMPEG
                    nome_arq_otimizado = os.path.join(PASTA_ALERTAS, f"QUEDA_ID{id_estabilizado}_{ts}.mp4")
                    print("⚙️ [FFMPEG] Comprimindo e otimizando vídeo para streaming mobile...")

                    comando_ffmpeg = f'ffmpeg -i "{nome_arq_bruto}" -c:v libx264 -crf 23 -pix_fmt yuv420p -movflags +faststart "{nome_arq_otimizado}" -y'
                    resultado_cmd = os.system(comando_ffmpeg)

                    if resultado_cmd == 0 and os.path.exists(nome_arq_otimizado):
                        os.remove(nome_arq_bruto)
                        video_para_enviar = nome_arq_otimizado
                    else:
                        print("⚠️ Falha na compressão do FFmpeg. Usando arquivo bruto...")
                        video_para_enviar = nome_arq_bruto

                    # 3. Despacha para a API usando o cliente injetado
                    api_client.enviar_deteccao(
                        postura="CAIDO",
                        client_id=1,
                        video_path=video_para_enviar,
                        info_adicional=f"Queda detectada para a pessoa com ID {id_estabilizado}.",
                        registrar_log_fn=registrar_log
                    )

        # Limpeza do dicionário de posições de rastreamento
        if tempo_atual % 5 < 0.1:
            posicoes_anteriores = {k: v for k, v in posicoes_anteriores.items() if (tempo_atual - v[2]) < 10}

        buffer_frames.append(frame.copy())

        # Atualização do painel administrativo de visualização
        cv2.putText(display_frame, f"FPS: {fps_real:.1f} | IDs Ativos: {len(ids_vivos_neste_frame)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Argos Gate - Protecao Residencial", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"❌ Erro em tempo de execução do loop principal: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()