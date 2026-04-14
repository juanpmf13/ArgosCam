import torch
import torch_directml
from ultralytics import YOLO

# 1. Configurar o dispositivo para GPU AMD
device = torch_directml.device()
print(f"🚀 Usando Dispositivo: {torch_directml.device_name(0)}")

# 2. Carregar o modelo
model = YOLO("/mnt/c/ArgosCam/modelos/versao3.pt")

# 3. Treinamento Adaptado para AMD
if __name__ == '__main__':
    model.train(
        data='/mnt/c/ArgosCam/scripts/argos_gate_gpu.yaml',
        epochs=80,          # Ajustado para completar o ciclo
        imgsz=640,
        batch=16,           # Pode tentar 32 se a placa tiver > 8GB VRAM
        device=device,
        workers=4,          # Aumentado para performance local
        project='/mnt/c/ArgosCam/Runs',
        name='ArgosGate_AMD_Rigor_V1',
        amp=False,


        # --- Parâmetros de Estabilidade ---
        multi_scale=False,
        overlap_mask=True,

        # --- Parâmetros de Rigor do Argos Gate ---
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