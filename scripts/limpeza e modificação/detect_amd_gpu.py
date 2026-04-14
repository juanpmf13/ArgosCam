import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from collections import deque

# --- 1. CONFIGURAÇÕES ---
ONNX_MODEL_PATH = r"C:\ArgosCam\Runs\ArgosGate_Final_Windows\weights\best.onnx"
VIDEO_PATH = r"C:\Users\juanp\OneDrive\Desktop\queda\veiajapa.mp4"
PASTA_ALERTAS = r"C:\ArgosCam\alertas"
PASTA_LOGS = r"C:\ArgosCam\logs"
COOLDOWN_SECONDS = 15

for p in [PASTA_ALERTAS, PASTA_LOGS]:
    if not os.path.exists(p): os.makedirs(p)

ARQUIVO_LOG = os.path.join(PASTA_LOGS, f"log_correcao_{time.strftime('%Y%m%d')}.txt")


def registrar_log(mensagem):
    with open(ARQUIVO_LOG, "a", encoding='utf-8') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {mensagem}\n")


def rodar_argos_otimizado():
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

    classes = ['em_pe', 'deitado', 'sentado', 'caindo', 'caido', 'agachado']
    historicos_por_id = {}
    posicoes_anteriores = {}
    proximo_id = 1
    ultimo_salvamento = 0

    # Buffer para o "Antes" (2 segundos a 30 FPS = 60 frames)
    buffer_antes = deque(maxlen=60)

    # Controle para o "Depois"
    gravando_depois = False
    frames_depois_contador = 0
    frames_depois_limite = 60  # 2 segundos depois
    video_writer = None

    cap = cv2.VideoCapture(VIDEO_PATH)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        tempo_atual = time.time()
        frame_visual = frame.copy()

        # Pré-processamento (Aumentamos um pouco a sensibilidade para 0.25 devido à oclusão)
        img_input = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)

        outputs = session.run(None, {'images': img_input})
        output = np.transpose(np.squeeze(outputs[0]))

        boxes, confs, class_ids = [], [], []
        for row in output:
            prob = row[4:].max()
            if prob > 0.10:  # Sensibilidade levemente maior
                class_id = row[4:].argmax()
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                bx = int((cx - w / 2) * W / 640)
                by = int((cy - h / 2) * H / 640)
                boxes.append([bx, by, int(w * W / 640), int(h * H / 640)])
                confs.append(float(prob))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confs, 0.25, 0.45)

        alerta_neste_frame = False
        if len(indices) > 0:
            for i in indices.flatten():
                bx, by, bw, bh = boxes[i]
                classe_nome = classes[class_ids[i]]
                centro = (bx + bw // 2, by + bh // 2)

                # Estabilização de ID
                id_estabilizado = None
                for old_id, dados in list(posicoes_anteriores.items()):
                    if np.sqrt((centro[0] - dados[0]) ** 2 + (centro[1] - dados[1]) ** 2) < 150:
                        id_estabilizado = old_id
                        break
                if id_estabilizado is None:
                    id_estabilizado = proximo_id
                    proximo_id += 1
                posicoes_anteriores[id_estabilizado] = (centro[0], centro[1], tempo_atual)

                # Lógica 7 de 10 (Corrigida para >= 7)
                if id_estabilizado not in historicos_por_id:
                    historicos_por_id[id_estabilizado] = deque(maxlen=10)

                historicos_por_id[id_estabilizado].append(classe_nome)
                queda_count = historicos_por_id[id_estabilizado].count('caindo') + \
                              historicos_por_id[id_estabilizado].count('caido')

                em_alerta = queda_count >= 7  # AGORA SIM 7 DE 10

                cor = (0, 0, 255) if em_alerta else (0, 255, 0)
                cv2.rectangle(frame_visual, (bx, by), (bx + bw, by + bh), cor, 2)
                cv2.putText(frame_visual, f"ID{id_estabilizado} {classe_nome}", (bx, by - 10), 0, 0.6, cor, 2)

                if em_alerta and (tempo_atual - ultimo_salvamento) > COOLDOWN_SECONDS and not gravando_depois:
                    registrar_log(f"🚨 QUEDA DETECTADA ID {id_estabilizado}")
                    ultimo_salvamento = tempo_atual

                    # Inicia gravação do Alerta
                    ts = time.strftime("%H%M%S")
                    caminho_video = os.path.join(PASTA_ALERTAS, f"ALERTA_COMPLETO_{ts}.mp4")
                    video_writer = cv2.VideoWriter(caminho_video, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

                    # Descarrega o "Antes" no vídeo
                    for f_antes in buffer_antes:
                        video_writer.write(f_antes)

                    gravando_depois = True
                    frames_depois_contador = 0

        # Gerenciamento da Gravação (Antes e Depois)
        if gravando_depois:
            video_writer.write(frame)
            frames_depois_contador += 1
            if frames_depois_contador >= frames_depois_limite:
                gravando_depois = False
                video_writer.release()
                print("✅ Vídeo de alerta salvo com sucesso (Antes + Depois).")
        else:
            buffer_antes.append(frame.copy())

        cv2.imshow("Argos Gate - Correcao Veia.mp4", frame_visual)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    if video_writer: video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    rodar_argos_otimizado()