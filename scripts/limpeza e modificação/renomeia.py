import os

# --- CONFIGURAÇÃO ---
base_dataset = r'C:\ArgosCam\dataset'
categorias = ['SEGURO', 'NAO_SEGURO']


def renomear_dataset_final():
    total_geral = 0

    for cat in categorias:
        caminho_cat = os.path.join(base_dataset, cat)
        if not os.path.exists(caminho_cat):
            continue

        for subpasta in os.listdir(caminho_cat):
            caminho_sub = os.path.join(caminho_cat, subpasta)

            if os.path.isdir(caminho_sub):
                print(f"Renomeando arquivos em: {cat}/{subpasta}...")

                # Lista e ordena para manter a sequência lógica
                arquivos = sorted(os.listdir(caminho_sub))

                for i, nome_antigo in enumerate(arquivos):
                    ext = os.path.splitext(nome_antigo)[1].lower()
                    if ext not in ['.jpg', '.jpeg', '.png']:
                        continue

                    # Novo nome: categoria_subpasta_numero.jpg
                    # Ex: SEGURO_EM_PE_00001.jpg
                    novo_nome = f"{cat}_{subpasta}_{i:06d}{ext}"

                    de = os.path.join(caminho_sub, nome_antigo)
                    para = os.path.join(caminho_sub, novo_nome)

                    # Evita erro se o arquivo já tiver o nome certo
                    if de != para:
                        os.rename(de, para)

                total_geral += len(arquivos)

    print(f"\n✅ Sucesso! {total_geral} arquivos foram padronizados.")


if __name__ == "__main__":
    renomear_dataset_final()