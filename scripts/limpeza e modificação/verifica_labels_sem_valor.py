import os
from PIL import Image

# Configurações de caminho
BASE_PATH = r"C:\ArgosCam\dataset_yolo"
FOLDERS = ["train", "val"]


def limpar_corrompidos():
    print("--- 🛡️ Iniciando Faxina no Argos Gate ---")

    total_removido = 0

    for folder in FOLDERS:
        img_dir = os.path.join(BASE_PATH, "images", folder)
        lbl_dir = os.path.join(BASE_PATH, "labels", folder)

        if not os.path.exists(img_dir):
            print(f"⚠️ Pasta não encontrada: {img_dir}")
            continue

        print(f"Scanning: {folder}...")

        for file in os.listdir(img_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(img_dir, file)
                # O label tem o mesmo nome da imagem, mas extensão .txt
                label_name = os.path.splitext(file)[0] + ".txt"
                label_path = os.path.join(lbl_dir, label_name)

                try:
                    # Tenta abrir e carregar os dados da imagem
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    print(f"❌ Corrompido detectado: {file}")

                    # 1. Remove a imagem
                    if os.path.exists(img_path):
                        os.remove(img_path)

                    # 2. Remove o label correspondente (se existir)
                    if os.path.exists(label_path):
                        os.remove(label_path)
                        print(f"   🗑️ Label pareado removido: {label_name}")
                    else:
                        print(f"   ℹ️ Nenhum label encontrado para esta imagem.")

                    total_removido += 1

        # 3. Limpeza de arquivos .cache (CRITICAL para o YOLO não bugar)
        for root, dirs, files in os.walk(BASE_PATH):
            for f in files:
                if f.endswith(".cache"):
                    try:
                        os.remove(os.path.join(root, f))
                        print(f"🧹 Cache limpo: {f}")
                    except:
                        pass

    print(f"\n--- ✅ Faxina concluída! Total de imagens removidas: {total_removido} ---")


if __name__ == "__main__":
    limpar_corrompidos()