import cv2
import os

path = r"C:\ArgosCam\dataset_yolo\images\val"

for f in os.listdir(path):
    full = os.path.join(path, f)
    img = cv2.imread(full)

    if img is None:
        print(f"❌ Imagem inválida: {f}")