import os
from PIL import Image
import cv2

# CONFIG
DATASET = r"C:\ArgosCam\dataset_yolo"
NC = 6  # número de classes

IMG_EXTS = [".jpg", ".jpeg", ".png"]

def is_image(file):
    return any(file.lower().endswith(ext) for ext in IMG_EXTS)

def validar_imagem(path):
    erros = []

    # PIL
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        erros.append(f"PIL erro: {e}")

    # OpenCV
    img = cv2.imread(path)
    if img is None:
        erros.append("OpenCV não leu")

    return erros

def validar_label(path):
    erros = []

    try:
        with open(path, "r") as f:
            linhas = f.readlines()

        if len(linhas) == 0:
            erros.append("Label vazio")

        for i, linha in enumerate(linhas):
            partes = linha.strip().split()

            if len(partes) != 5:
                erros.append(f"Linha {i}: formato inválido")
                continue

            try:
                cls = int(partes[0])
                coords = list(map(float, partes[1:]))

                if cls < 0 or cls >= NC:
                    erros.append(f"Linha {i}: classe fora do range ({cls})")

                for c in coords:
                    if c < 0 or c > 1:
                        erros.append(f"Linha {i}: bbox fora do range")

            except:
                erros.append(f"Linha {i}: valor não numérico")

    except Exception as e:
        erros.append(f"Erro ao ler label: {e}")

    return erros

def validar_split(split):
    print(f"\n=== VALIDANDO {split.upper()} ===")

    img_dir = os.path.join(DATASET, "images", split)
    lbl_dir = os.path.join(DATASET, "labels", split)

    imgs = os.listdir(img_dir)
    lbls = os.listdir(lbl_dir)

    # 1. Arquivos inválidos
    for f in imgs:
        if not is_image(f):
            print(f"❌ Arquivo inválido em images: {f}")

    # 2. Checar imagens
    for img in imgs:
        if not is_image(img):
            continue

        img_path = os.path.join(img_dir, img)
        name = os.path.splitext(img)[0]
        lbl_path = os.path.join(lbl_dir, name + ".txt")

        # imagem abre?
        erros_img = validar_imagem(img_path)
        if erros_img:
            print(f"❌ Imagem problemática: {img} -> {erros_img}")

        # label existe?
        if not os.path.exists(lbl_path):
            print(f"❌ Sem label: {img}")
        else:
            erros_lbl = validar_label(lbl_path)
            if erros_lbl:
                print(f"❌ Label problemática: {name}.txt -> {erros_lbl}")

    # 3. labels sem imagem
    for lbl in lbls:
        name = os.path.splitext(lbl)[0]

        if not any(os.path.exists(os.path.join(img_dir, name + ext)) for ext in IMG_EXTS):
            print(f"❌ Label sem imagem: {lbl}")

def main():
    validar_split("train")
    validar_split("val")

    print("\n✅ Validação concluída")

if __name__ == "__main__":
    main()