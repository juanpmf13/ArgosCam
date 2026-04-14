import numpy as np
import cv2
import os
import time
from ultralytics import YOLO
from collections import deque

# --- 1. CONFIGURAÇÕES ---
FONTE_ARQUIVO_LOCAL = r"C:\Users\juanp\OneDrive\Desktop\queda\veio.mp4"
LINK_CELULAR = "http://10.0.10.239:8080/video"

PATH_MODELO = r"C:\ArgosCam\Runs\ArgosGate_Final_Windows\weights\best.pt"
PASTA_ALERTAS = r"C:\ArgosCam\alertas"
PASTA_LOGS = r"C:\ArgosCam\logs"
COOLDOWN_SECONDS = 15

# Criar pastas
for p in [PASTA_ALERTAS, PASTA_LOGS]:
    if not os.path.exists(p): os.makedirs(p)

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


# --- 2. INICIALIZAÇÃO ---
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

# --- 3. LOOP PRINCIPAL ---
try:
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if modo_streaming:
                time.sleep(2);
                cap = cv2.VideoCapture(LINK_CELULAR);
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

        # IA - Aumentamos o rigor (conf) para evitar IDs fantasmas
        results = modelo.track(
            frame,
            imgsz=640,
            persist=True,
            conf=0.40,  # Baixe para 0.40 (no vídeo, a detecção some porque a conf cai)
            iou=0.4,  # Ajuda a manter IDs separados
            tracker="botsort.yaml",
            augment=True  # Faz a IA 'tentar mais uma vez' antes de desistir da caixa
        )

        ids_vivos_neste_frame = []

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box, obj_id, cls in zip(boxes, ids, clss):
                classe_nome = results[0].names[cls]
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Centro da pessoa

                # --- LÓGICA DE ESTABILIZAÇÃO DE ID ---
                # Se o tracker gerou um ID novo (ex: 15), verificamos se há um ID antigo
                # que sumiu perto desta posição nos últimos 3 segundos.
                id_estabilizado = obj_id
                for id_antigo, dados in list(posicoes_anteriores.items()):
                    velho_x, velho_y, velho_ts = dados
                    distancia = np.sqrt((cx - velho_x) ** 2 + (cy - velho_y) ** 2)

                    if distancia < 100 and (tempo_atual - velho_ts) < 3.0:
                        id_estabilizado = id_antigo  # Mantém o ID original
                        break

                posicoes_anteriores[id_estabilizado] = (cx, cy, tempo_atual)
                ids_vivos_neste_frame.append(id_estabilizado)

                # Gerenciamento de Histórico com ID Estabilizado
                if id_estabilizado not in historicos_por_id:
                    historicos_por_id[id_estabilizado] = deque(maxlen=60)  # Aumentado para 40

                historicos_por_id[id_estabilizado].append(classe_nome)

                # Lógica de Alerta (Mais rigorosa: precisa de 25 frames de queda)
                alert_count = historicos_por_id[id_estabilizado].count('caindo') + \
                              historicos_por_id[id_estabilizado].count('caido')

                # Desenho
                vx1, vy1, vx2, vy2 = map(int, box * proporcao)
                cor = (0, 0, 255) if alert_count > 10 else (0, 255, 0)
                cv2.rectangle(display_frame, (vx1, vy1), (vx2, vy2), cor, 2)
                cv2.putText(display_frame, f"PESSOA {id_estabilizado}: {classe_nome}", (vx1, vy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

                # Disparo de Alerta
                if alert_count >= 25 and (tempo_atual - ultimo_salvamento) > COOLDOWN_SECONDS:
                    ts = time.strftime("%H%M%S")
                    nome_arq = os.path.join(PASTA_ALERTAS, f"QUEDA_ID{id_estabilizado}_{ts}.mp4")
                    print(f"🚨 QUEDA CONFIRMADA! ID {id_estabilizado}")

                    out = cv2.VideoWriter(nome_arq, cv2.VideoWriter_fourcc(*'mp4v'), FPS_VIDEO, (WIDTH, HEIGHT))
                    for f_buf in buffer_frames: out.write(f_buf)
                    out.release()
                    ultimo_salvamento = tempo_atual

        # Limpeza de IDs muito antigos do cache de posições
        if tempo_atual % 5 < 0.1:  # Limpa a cada 5 segundos
            posicoes_anteriores = {k: v for k, v in posicoes_anteriores.items() if (tempo_atual - v[2]) < 10}

        buffer_frames.append(frame.copy())

        # Dashboard de Status
        cv2.putText(display_frame, f"FPS: {fps_real:.1f} | IDs Ativos: {len(ids_vivos_neste_frame)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Argos Gate - Protecao Residencial", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f"❌ Erro: {e}")
finally:
    cap.release();
    cv2.destroyAllWindows()