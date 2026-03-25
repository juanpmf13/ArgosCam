from ultralytics import YOLO

def treinar_modelo():
    model = YOLO("yolo11n.pt")

    model.train(
        data=r"C:\ArgosCam\scripts\argos_gate.yaml",
        epochs=5,
        imgsz=640,
        batch=8,  # Reduzimos para 8 para diminuir a pressão na RAM
        device="cpu",
        optimizer='SGD',  # Mudamos para SGD (é o mais simples e robusto de todos)
        lr0=0.0001,  # Taxa de aprendizado mínima
        warmup_epochs=0,  # Desativa warmup para evitar saltos iniciais
        amp=False,
        plots=True,
        cache=False,  # FORÇA a não usar cache (lê direto do disco)
        workers=0,  # FORÇA o Python a usar apenas a thread principal (evita erro de leitura paralela)
        exist_ok=True
    )

if __name__ == "__main__":
    treinar_modelo()