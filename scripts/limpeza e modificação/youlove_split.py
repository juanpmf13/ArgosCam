import os
import shutil
import random

# --- CONFIGURAÇÃO ---
origem_raiz = r'C:\ArgosCam\dataset'
destino_raiz = r'C:\ArgosCam\dataset_yolo'
split_ratio = 0.8  # 80% para treino, 20% para validação

categorias = {
    'SEGURO/EM_PE': 0,
    'SEGURO/AGACHADO': 1,
    'SEGURO/SENTADO': 2,
    'SEGURO/DEITADO': 3,
    'NAO_SEGURO/CAINDO': 4,
    'NAO_SEGURO/CAIDO': 5
}


def preparar_pastas():
    for fase in ['train', 'val']:
        os.makedirs(os.path.join(destino_raiz, 'images', fase), exist_ok=True)
        os.makedirs(os.path.join(destino_raiz, 'labels', fase), exist_ok=True)


def mover_e_organizar():
    preparar_pastas()

    for subpath, class_id in categorias.items():
        caminho_cat = os.path.join(origem_raiz, subpath)
        if not os.path.exists(caminho_cat): continue

        arquivos = [f for f in os.listdir(caminho_cat) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(arquivos)  # Embaralha para o treino não viciar na ordem dos vídeos

        limite = int(len(arquivos) * split_ratio)
        train_files = arquivos[:limite]
        val_files = arquivos[limite:]

        for lista, fase in [(train_files, 'train'), (val_files, 'val')]:
            print(f"Movendo {len(lista)} arquivos de {subpath} para {fase}...")
            for img_nome in lista:
                # Caminho da imagem
                src_img = os.path.join(caminho_cat, img_nome)
                dst_img = os.path.join(destino_raiz, 'images', fase, img_nome)
                shutil.copy2(src_img, dst_img)  # Usamos copy para não deletar sua origem por segurança

    print(f"\n✅ Estrutura criada em: {destino_raiz}")


if __name__ == "__main__":
    mover_e_organizar()