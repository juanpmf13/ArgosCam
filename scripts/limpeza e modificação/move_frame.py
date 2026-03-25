import os
import shutil

# --- CONFIGURAÇÕES ---
base_destino = r'C:\ArgosCam\dataset'

# DEFINA OS INTERVALOS (Frame Inicial, Frame Final)
intervalos = {
    'SEGURO/EM_PE': (),
    'SEGURO/AGACHADO': (),
    'SEGURO/SENTADO': (),
    'SEGURO/DEITADO': (),
    'NAO_SEGURO/CAINDO': (),
    'NAO_SEGURO/CAIDO': ()
}


def distribuir_frames(caminho_origem):
    # Garante que as pastas de destino existam
    for subpath in intervalos.keys():
        os.makedirs(os.path.join(base_destino, subpath), exist_ok=True)

    if not os.path.exists(caminho_origem):
        print(f"⚠️ Pasta não encontrada: {caminho_origem}")
        return

    arquivos = sorted(os.listdir(caminho_origem))
    nome_pasta_video = os.path.basename(caminho_origem)

    # Contador para a numeração sequencial dentro desta execução
    contador = 0

    for arquivo in arquivos:
        try:
            # Extrai o número do frame original para verificar o intervalo
            num_frame = int(''.join(filter(str.isdigit, arquivo)))
        except ValueError:
            continue

        for subpath, faixa in intervalos.items():
            if faixa and faixa[0] <= num_frame <= faixa[1]:
                origem = os.path.join(caminho_origem, arquivo)

                # NOVO NOME: nome_da_pasta_numero_sequencial.jpg
                # Exemplo: cam1_00001.jpg
                extensao = os.path.splitext(arquivo)[1]
                novo_nome = f"{nome_pasta_video}_{contador:05d}{extensao}"

                destino = os.path.join(base_destino, subpath, novo_nome)

                # Verifica se o arquivo já existe para não sobrescrever caso rode o script 2x
                if os.path.exists(destino):
                    novo_nome = f"{nome_pasta_video}_{contador:05d}_v2{extensao}"
                    destino = os.path.join(base_destino, subpath, novo_nome)

                shutil.copy2(origem, destino)
                contador += 1
                break

    print(f"✅ Vídeo '{nome_pasta_video}': {contador} frames movidos com sucesso!")


if __name__ == "__main__":
    # Loop para percorrer de cam1 até cam8
    for n in range(1, 41):
        pasta_cam = r'C:\ArgosCam\dataset_extraido_limpo\fall-0'+str(n)+'-cam0-rgb'
        distribuir_frames(pasta_cam)
        pasta_cam = r'C:\ArgosCam\dataset_extraido_limpo\fall-0' + str(n) + '-cam1-rgb'
        distribuir_frames(pasta_cam)