import os
from ultralytics import YOLO

def treinar_modelo():
    # --- AJUSTE DE CAMINHOS PARA O DISCO D ---
    caminho_modelo_base = r"D:\ArgosCam\Runs\ArgosGate_D_Drive\weights\last.pt"
    caminho_data_yaml = r"D:\ArgosCam\scripts\argos_gate.yaml"
    diretorio_projeto = r"D:\ArgosCam\Runs"

    # 2. Carregar o modeloq
    if os.path.exists(caminho_modelo_base):
        model = YOLO(caminho_modelo_base)
        print(f"✅ Carregando progresso do Disco D: {caminho_modelo_base}")
    else:
        # Caso o treino novo ainda não tenha gerado o last.pt, tenta o anterior
        caminho_backup = r"D:\ArgosCam\Runs\ArgosGate_Final_Windows\weights\last.pt"
        if os.path.exists(caminho_backup):
            model = YOLO(caminho_backup)
            print("⚠️ Usando backup da pasta anterior.")
        else:
            model = YOLO("yolo11n.pt")
            print("🆕 Começando do zero com yolo11n.pt")

    # 3. Execução do Treino
    model.train(
        data=caminho_data_yaml,
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=r"D:\ArgosCam\Runs",  # <--- FORÇA O DISCO D AQUI
        name='ArgosGate_D_Drive',  # <--- NOME DA PASTA NO D:
        resume=True,  # <--- MUDE PARA FALSE PARA ELE ACEITAR O NOVO PROJECT
        exist_ok=True,  # Permite escrever na pasta se ela já existir
        # --- CONFIGURAÇÕES DE ESTABILIDADE ---
        amp=True,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        plots=True,
        # --- PARÂMETROS DE RIGOR ---
        cls=2.5,
        box=10.0,
        mixup=0.15,
        close_mosaic=20,
        degrees=15.0,
        scale=0.5,
        fliplr=0.5
    )

    # 4. EXPORTAÇÃO
    print("🚀 Treino finalizado! Exportando para ONNX...")
    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)

if __name__ == "__main__":
    treinar_modelo()