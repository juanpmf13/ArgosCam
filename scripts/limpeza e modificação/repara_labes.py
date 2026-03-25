import os
from tqdm import tqdm

# Mapeamento oficial do seu argos_gate.yaml (Imagem image_669c91.png)
mapping = {
    "em_pe": 0,
    "agachado": 1,
    "sentado": 2,
    "deitado": 3,
    "caindo": 4,
    "caido": 5
}

# Caminho onde estão os seus 65.230 arquivos .txt
label_path = r'/dataset_yolo/labels/train'


def fix_labels():
    files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    print(f"🔧 Reparando {len(files)} labels...")

    for filename in tqdm(files):
        path = os.path.join(label_path, filename)

        # Identificar a classe pelo nome do arquivo
        target_class = None
        for key, value in mapping.items():
            if key in filename.lower():
                target_class = value
                break

        if target_class is None:
            continue

        # Ler o conteúdo, trocar o primeiro número e salvar
        with open(path, 'r') as f:
            lines = f.readlines()

        with open(path, 'w') as f:
            for line in lines:
                parts = line.split()
                if len(parts) > 0:
                    # Substitui o ID antigo (0) pelo ID correto do mapeamento
                    parts[0] = str(target_class)
                    f.write(" ".join(parts) + "\n")


if __name__ == "__main__":
    fix_labels()