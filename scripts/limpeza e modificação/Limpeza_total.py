import os
from PIL import Image

BASE_PATH = r"C:\ArgosCam\dataset_yolo"
FOLDERS = ["train", "val"]


def limpar_dataset():
    print("--- 🛡️ Iniciando Limpeza Profunda no Argos Gate ---")
    removidos = 0

    for folder in FOLDERS:
        img_dir = os.path.join(BASE_PATH, "images", folder)
        lbl_dir = os.path.join(BASE_PATH, "labels", folder)

        if not os.path.exists(img_dir): continue

        for file in os.listdir(img_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(img_dir, file)
                label_path = os.path.join(lbl_dir, os.path.splitext(file)[0] + ".txt")
                corrompido = False

                # 1. Validar Imagem
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except:
                    corrompido = True

                # 2. Validar Labels (Coordenadas YOLO 0 a 1)
                if not corrompido and os.path.exists(label_path):
                    try:
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.split()
                                if len(parts) != 5:
                                    corrompido = True;
                                    break
                                vals = [float(x) for x in parts[1:]]
                                if any(v < 0 or v > 1 for v in vals):
                                    corrompido = True;
                                    break
                    except:
                        corrompido = True

                if corrompido:
                    if os.path.exists(img_path): os.remove(img_path)
                    if os.path.exists(label_path): os.remove(label_path)
                    removidos += 1

    # 3. Limpar Caches (Obrigatório)
    for root, dirs, files in os.walk(BASE_PATH):
        for f in files:
            if f.endswith(".cache"):
                os.remove(os.path.join(root, f))

    print(f"✅ Faxina concluída! {removidos} pares inválidos removidos.")


if __name__ == "__main__":
    limpar_dataset()