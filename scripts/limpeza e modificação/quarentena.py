import os
import shutil
from tqdm import tqdm


def organizar_dataset():
    # Caminhos originais
    base_path = r'C:\ArgosCam\dataset_yolo'
    # Pasta de destino para imagens sem labels
    quarentena_path = r'C:\ArgosCam\quarentena'

    etapas = ['train', 'val']
    extensoes_img = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for etapa in etapas:
        img_dir = os.path.join(base_path, 'images', etapa)
        lbl_dir = os.path.join(base_path, 'labels', etapa)

        # Criar pastas de quarentena correspondentes
        destino_img = os.path.join(quarentena_path, 'images', etapa)
        os.makedirs(destino_img, exist_ok=True)

        if not os.path.exists(img_dir):
            print(f"⚠️ Pasta {img_dir} não encontrada. Pulando...")
            continue

        # Listar arquivos
        arquivos_img = [f for f in os.listdir(img_dir) if f.lower().endswith(extensoes_img)]
        labels_existentes = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')}

        print(f"\n📦 Analisando {etapa.upper()}...")

        movidos = 0
        for nome_img in tqdm(arquivos_img):
            nome_base = os.path.splitext(nome_img)[0]

            # Se não existe label para esta imagem, movemos para quarentena
            if nome_base not in labels_existentes:
                origem = os.path.join(img_dir, nome_img)
                destino = os.path.join(destino_img, nome_img)

                try:
                    shutil.move(origem, destino)
                    movidos += 1
                except Exception as e:
                    print(f"❌ Erro ao mover {nome_img}: {e}")

        print(f"✅ Sucesso: {movidos} imagens movidas para {destino_img}")


if __name__ == "__main__":
    organizar_dataset()