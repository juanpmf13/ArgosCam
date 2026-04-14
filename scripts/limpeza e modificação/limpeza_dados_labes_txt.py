import os


def limpar_labels(label_path):
    print(f"🧹 Corrigindo labels em: {label_path}")
    for file in os.listdir(label_path):
        if file.endswith(".txt"):
            filepath = os.path.join(label_path, file)
            with open(filepath, 'r') as f:
                lines = f.readlines()

            novas_linhas = []
            for line in lines:
                parts = line.split()
                # Mantém apenas as 5 primeiras colunas (ID, X, Y, W, H)
                if len(parts) >= 5:
                    novas_linhas.append(" ".join(parts[:5]))

            with open(filepath, 'w') as f:
                f.write("\n".join(novas_linhas))


# Execute para as duas pastas
limpar_labels(r"C:\ArgosCam\dataset_yolo\labels\train")
limpar_labels(r"C:\ArgosCam\dataset_yolo\labels\val")
print("✅ Labels simplificados! Tente rodar o treino_CPU.py novamente.")