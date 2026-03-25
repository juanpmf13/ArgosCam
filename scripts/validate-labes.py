import os
import cv2
import random

# --- CONFIGURAÇÕES ---
caminho_imagens = r'C:\ArgosCam\dataset_yolo\images\train'
caminho_labels = r'C:\ArgosCam\dataset_yolo\labels\train'


def validar_amostra():
    # Listar apenas imagens que já possuem labels gerados
    labels_gerados = [f.replace('.txt', '') for f in os.listdir(caminho_labels) if f.endswith('.txt')]

    if not labels_gerados:
        print("❌ Nenhum label encontrado para validar ainda.")
        return

    # Escolher uma imagem aleatória das que já foram processadas
    nome_base = random.choice(labels_gerados)
    img_path = os.path.join(caminho_imagens, nome_base + ".jpg")  # Ajuste se for .png
    label_path = os.path.join(caminho_labels, nome_base + ".txt")

    if not os.path.exists(img_path):
        print(f"⚠️ Imagem {nome_base} não encontrada (verifique a extensão).")
        return

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(label_path, 'r') as f:
        for linha in f:
            dados = linha.split()
            cls = dados[0]
            # YOLO format: class x_center y_center width height (normalizado)
            x_cnt, y_cnt, w_box, h_box = map(float, dados[1:])

            # Converter de normalizado para pixels da imagem original
            # O script anterior normalizou por 640 (tamanho da entrada do ONNX)
            # Então multiplicamos pelo tamanho real da imagem
            x1 = int((x_cnt - w_box / 2) * w)
            y1 = int((y_cnt - h_box / 2) * h)
            x2 = int((x_cnt + w_box / 2) * w)
            y2 = int((y_cnt + h_box / 2) * h)

            # Desenhar o retângulo (Verde para pessoa)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Classe: {cls}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"🖼️ Exibindo validação para: {nome_base}")
    cv2.imshow("Validacao ArgosCam", img)
    print("Pressione qualquer tecla para fechar ou feche a janela.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    validar_amostra()