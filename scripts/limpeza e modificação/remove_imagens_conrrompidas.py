import os
from PIL import Image

# Caminho base do seu dataset
base_path = r"C:\ArgosCam\dataset_yolo"


def limpar_dataset(subfolder):
    img_dir = os.path.join(base_path, "images", subfolder)
    lbl_dir = os.path.join(base_path, "labels", subfolder)

    removidos = 0

    if not os.path.exists(img_dir):
        return

    for file in os.listdir(img_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(img_dir, file)
            label_path = os.path.join(lbl_dir, file.rsplit('.', 1)[0] + ".txt")

            try:
                # Tenta abrir e carregar os pixels reais
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                print(f"🔥 Corrompido: {file}. Removendo imagem e label...")

                # Remove a imagem
                if os.path.exists(img_path):
                    os.remove(img_path)

                # Remove o label correspondente para manter o par
                if os.path.exists(label_path):
                    os.remove(label_path)

                removidos += 1

    print(f"✅ Finalizado em {subfolder}: {removidos} arquivos removidos.")


# Executa para as duas pastas
limpar_dataset("train")
limpar_dataset("val")