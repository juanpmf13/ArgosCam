import os

# --- CONFIGURAÇÃO ---
diretorio_raiz = r'C:\ArgosCam\dataset_extraido_limpo'


def renomear_frames_nas_subpastas():
    # Pega apenas as pastas dentro do diretório raiz
    subpastas = [f for f in os.listdir(diretorio_raiz) if os.path.isdir(os.path.join(diretorio_raiz, f))]

    if not subpastas:
        print("Nenhuma subpasta encontrada no diretório especificado.")
        return

    print(f"Iniciando renomeação em {len(subpastas)} pastas...\n")

    for pasta in subpastas:
        caminho_completo_pasta = os.path.join(diretorio_raiz, pasta)

        # Lista arquivos, garantindo que pegamos apenas imagens e em ordem alfabética/numérica
        arquivos = sorted([f for f in os.listdir(caminho_completo_pasta)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        print(f"Processando [{pasta}]: {len(arquivos)} arquivos encontrados.")

        for i, nome_antigo in enumerate(arquivos):
            extensao = os.path.splitext(nome_antigo)[1]

            # Formato: cam1_00001.jpg
            novo_nome = f"{pasta}_{i:05d}{extensao}"

            caminho_antigo = os.path.join(caminho_completo_pasta, nome_antigo)
            caminho_novo = os.path.join(caminho_completo_pasta, novo_nome)

            # Renomeia (com tratamento de erro básico)
            try:
                if caminho_antigo != caminho_novo:
                    os.rename(caminho_antigo, caminho_novo)
            except Exception as e:
                print(f"Erro ao renomear {nome_antigo}: {e}")

    print("\n✅ Concluído! Todos os frames foram renomeados e padronizados.")


if __name__ == "__main__":
    renomear_frames_nas_subpastas()