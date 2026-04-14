import cv2
import os
import time
from ultralytics import YOLO
from collections import deque

# --- CONFIGURAÇÕES ---
PATH_MODELO = r"/modelos/versao2.pt"
VIDEO_FONTE = r"C:\ArgosCam\teste.mp4"
PASTA_ALERTAS = r"C:\ArgosCam\alertas"
COOLDOWN_SECONDS = 5

if not os.path.exists(PASTA_ALERTAS):
    os.makedirs(PASTA_ALERTAS)

# Buffer de 150 frames (~5-7 segundos) para o vídeo de alerta
buffer_frames_processados = deque(maxlen=150)
historicos_por_id = {}
ultimo_salvamento = 0

# Carregando modelo na CPU (Estabilidade total)
model = YOLO(PATH_MODELO)

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_saida = 20

print(f"--- 🛡️ Argos Gate: MODO SENSIBILIDADE ALTA ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    tempo_atual = time.time()

    # tracker='bytetrack.yaml' ajuda a não perder o ID em movimentos rápidos
    # conf=0.25 para detectar mesmo que o modelo esteja na dúvida
    results = model.track(frame, persist=True, verbose=False,
                          conf=0.25, device="cpu", tracker="bytetrack.yaml")

    display_frame = frame.copy()

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        confiancas = results[0].boxes.conf.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box, obj_id, cls, conf in zip(boxes, ids, clss, confiancas):
            classe_nome = results[0].names[cls]

            # Histórico menor (10) para reagir mais rápido a mudanças
            if obj_id not in historicos_por_id:
                historicos_por_id[obj_id] = deque(maxlen=10)
            historicos_por_id[obj_id].append(classe_nome)

            # Print técnico no console para diagnóstico
            if classe_nome in ['em queda', 'caido']:
                print(f"🔍 [SCAN] ID:{obj_id} detectado como {classe_nome} | Conf: {conf:.2f}")

            # Cores: Alerta em Vermelho, Seguro em Verde
            cor = (0, 0, 255) if classe_nome in ['em queda', 'caido'] else (0, 255, 0)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), cor, 2)

            label_texto = f"ID:{obj_id} {classe_nome} ({conf:.2f})"
            cv2.putText(display_frame, label_texto, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

            # GATILHO: Se em 10 frames, pelo menos 3 forem queda/caído, ele dispara
            alert_count = historicos_por_id[obj_id].count('em queda') + historicos_por_id[obj_id].count('caido')

            if alert_count >= 3 and (tempo_atual - ultimo_salvamento) > COOLDOWN_SECONDS:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                nome_arquivo = os.path.join(PASTA_ALERTAS, f"QUEDA_CONFIRMADA_ID{obj_id}_{timestamp}.mp4")

                print(f"🚨 >>> DISPARANDO ALERTA GRAVADO: {nome_arquivo}")

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(nome_arquivo, fourcc, fps_saida, (frame_width, frame_height))

                for f_salvar in buffer_frames_processados:
                    out.write(f_salvar)
                out.release()

                ultimo_salvamento = tempo_atual
                historicos_por_id[obj_id].clear()  # Limpa para não repetir o alerta no mesmo segundo

    # Adiciona o frame com os desenhos no buffer
    buffer_frames_processados.append(display_frame.copy())

    # Feedback de Cooldown na tela
    tempo_restante = COOLDOWN_SECONDS - (tempo_atual - ultimo_salvamento)
    if tempo_restante > 0:
        cv2.putText(display_frame, f"SISTEMA EM COOLDOWN: {tempo_restante:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Argos Gate - Analise de Sensibilidade", display_frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()