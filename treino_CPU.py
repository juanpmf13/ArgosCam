import os
from ultralytics import YOLO

def treinar_modelo():
    # 1. Caminhos no formato Windows (Use barras normais / ou duplas \\)
    caminho_modelo_base = r"C:\ArgosCam\Runs\ArgosGate_Final_Windows\weights\last.pt"
    caminho_data_yaml = r"C:\ArgosCam\scripts\argos_gate.yaml"
    diretorio_projeto = r"C:\ArgosCam\Runs"

    # 2. Carregar o modelo
    # Se o arquivo versao3.pt não existir, ele baixará o yolo11n.pt automaticamente
    if os.path.exists(caminho_modelo_base):
        model = YOLO(caminho_modelo_base)
        print("✅ Carregando progresso anterior (Versão 3)")
    else:
        model = YOLO("yolo11n.pt")
        print("⚠️ Versão 3 não encontrada. Começando do yolo11n base.")

    # 3. Execução do Treino
    # No Windows, 'device' pode ser 'cpu' para garantir que termine sem erros de driver
    model.train(
        model=r'C:\ArgosCam\Runs\ArgosGate_Final_Windows2\weights\last.pt',  # <--- AQUI ESTÁ O SEGREDO
        resume=True,
        data=caminho_data_yaml,
        epochs=80,
        imgsz=640,
        batch=8,
        device='cpu',
        workers=0,
        project=diretorio_projeto,
        name='ArgosGate_Final_Windows',

        # --- OS CORRETIVOS PARA O ERRO 'NAN' ---
        amp=False,  # Garantir que está desligado
        lr0=0.001,  # Diminuir a velocidade de aprendizado inicial
        lrf=0.01,  # Diminuir a velocidade final
        warmup_epochs=3.0,  # Dar 3 épocas de "aquecimento" lento para o modelo
        plots=True,  # Gerar gráficos para vermos onde o erro acontece

        # --- Parâmetros de Rigor (Mantidos) ---
        cls=2.5,
        box=10.0,
        mixup=0.15,
        close_mosaic=20,
        degrees=15.0,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )

    # 4. EXPORTAÇÃO PARA ONNX (A mágica para rodar na GPU AMD depois)
    print("🚀 Treino finalizado! Exportando para ONNX para rodar na GPU AMD...")
    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)

if __name__ == "__main__":
    # Proteção necessária para Multiprocessing no Windows
    treinar_modelo()