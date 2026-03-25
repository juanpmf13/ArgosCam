import os

def verificar_integridade(base_path):
    pastas = ['train', 'val']
    extensoes_imagem = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    print(f"--- Relatório de Integridade: {base_path} ---")

    for etapa in pastas:
        img_dir = os.path.join(base_path, 'images', etapa)
        lbl_dir = os.path.join(base_path, 'labels', etapa)

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"⚠️ Erro: Pasta de {etapa} não encontrada em {base_path}")
            continue

        # Lista nomes de arquivos sem extensão
        imagens = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(extensoes_imagem)}
        labels = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')}

        # 1. Imagens que não possuem arquivo TXT
        sem_label = imagens - labels
        # 2. Arquivos TXT que não possuem imagem correspondente (lixo no dataset)
        sem_imagem = labels - imagens

        print(f"\n📂 Resultados para [{etapa.upper()}]:")
        print(f"   - Total de Imagens: {len(imagens)}")
        print(f"   - Total de Labels: {len(labels)}")

        if sem_label:
            print(f"   ❌ ERRO: {len(sem_label)} imagens SEM arquivo .txt!")
            # Mostra apenas as 5 primeiras para não poluir o console
            print(f"   Exemplos: {list(sem_label)[:5]}")
        else:
            print(f"   ✅ Todas as imagens têm labels.")

        if sem_imagem:
            print(f"   ⚠️ AVISO: {len(sem_imagem)} arquivos .txt extras (sem imagem correspondente).")
            print(f"   Exemplos: {list(sem_imagem)[:5]}")

# Configure o caminho raiz do seu dataset aqui
caminho_dataset = r'C:\ArgosCam\dataset_yolo'
verificar_integridade(caminho_dataset)