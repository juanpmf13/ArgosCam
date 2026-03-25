import os
from tqdm import tqdm

# Mapeamento conforme seu argos_gate.yaml (Imagem image_669c91.png)
mapping = {
    "em_pe": 0,
    "agachado": 1,
    "sentado": 2,
    "deitado": 3,
    "caindo": 4,
    "caido": 5
}

# Caminhos das suas pastas de labels
train_path = r'/dataset_yolo/labels/train'
# Se você tiver uma pasta val, adicione aqui também
paths_to_fix = [train_path]


def reparar_classes_por_nome():
    for folder in paths_to_fix:
        if not os.path.exists(folder): continue

        files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        print(f"🔧 Corrigindo classes em: {folder}")

        for filename in tqdm(files):
            full_path = os.path.join(folder, filename)

            # Descobrir qual a classe correta baseada no nome do arquivo
            target_id = None
            for label_name, label_id in mapping.items():
                if label_name in filename.lower():
                    target_id = label_id
                    break

            if target_id is None: continue

            # Abrir o arquivo, trocar o primeiro número e salvar
            with open(full_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) > 0:
                    parts[0] = str(target_id)  # Troca o 0 pelo ID correto (ex: 5)
                    new_lines.append(" ".join(parts))

            with open(full_path, 'w') as f:
                f.write("\n".join(new_lines))


if __name__ == "__main__":
    reparar_classes_por_nome()