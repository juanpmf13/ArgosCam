import os
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
pastas_para_processar = [
    {
        "imagens": r'C:\ArgosCam\dataset_yolo\images\train',
        "labels": r'C:\ArgosCam\dataset_yolo\labels\train'
    },
    {
        "imagens": r'C:\ArgosCam\dataset_yolo\images\val',
        "labels": r'C:\ArgosCam\dataset_yolo\labels\val'
    }
]

modelo_onnx = 'yolo11n.onnx'

def definir_classe(nome_arquivo):
    nome_low = nome_arquivo.lower()
    if 'caindo' in nome_low: return 3
    if 'caido' in nome_low: return 4
    if 'em_pe' in nome_low: return 0
    if 'deitado' in nome_low: return 1
    if 'sentado' in nome_low: return 2
    return 0

# Configuração da Sessão ONNX (DirectML para sua RX 6750 XT)
providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(modelo_onnx, providers=providers)

def processar_tudo():
    for config in pastas_para_processar:
        img_dir = config["imagens"]
        lbl_dir = config["labels"]

        os.makedirs(lbl_dir, exist_ok=True)

        imagens = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\n📂 Processando pasta: {os.path.basename(img_dir)}")

        for nome_img in tqdm(imagens, desc="Detectando"):
            caminho_img = os.path.join(img_dir, nome_img)
            nome_base = os.path.splitext(nome_img)[0]
            caminho_txt = os.path.join(lbl_dir, f"{nome_base}.txt")

            if os.path.exists(caminho_txt):
                continue

            # --- A PARTIR DAQUI TUDO DEVE ESTAR IDENTADO DENTRO DO FOR ---
            img_original = cv2.imread(caminho_img)
            if img_original is None:
                continue

            # Pré-processamento
            img_input = cv2.resize(img_original, (640, 640))
            img_input = img_input.transpose(2, 0, 1)
            img_input = np.expand_dims(img_input, axis=0).astype(np.float32) / 255.0

            # Inferência
            outputs = session.run(None, {session.get_inputs()[0].name: img_input})
            output = outputs[0][0].T  # [8400, 84]

            boxes = []
            confidences = []

            for row in output:
                scores = row[4:]
                class_id = np.argmax(scores)
                conf = scores[class_id]

                if conf > 0.4 and class_id == 0: # 0 = Pessoa no modelo base
                    xc, yc, w, h = row[:4]
                    x1 = xc - (w / 2)
                    y1 = yc - (h / 2)
                    boxes.append([float(x1), float(y1), float(w), float(h)])
                    confidences.append(float(conf))

            # NMS para evitar duplicatas
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.45)

            if len(indices) > 0:
                classe_alvo = definir_classe(nome_img)
                with open(caminho_txt, 'w') as f:
                    # Garantir compatibilidade com diferentes versões do OpenCV
                    for i in indices.flatten():
                        box = boxes[i]

                        # Normalização YOLO (0-1)
                        xn = (box[0] + (box[2] / 2)) / 640
                        yn = (box[1] + (box[3] / 2)) / 640
                        wn = box[2] / 640
                        hn = box[3] / 640

                        f.write(f"{classe_alvo} {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}\n")

if __name__ == "__main__":
    processar_tudo()
    print("\n✅ Processamento completo para Train e Val!")