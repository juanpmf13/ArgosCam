import os
import sys

# --- CONFIGURAÇÃO DE AMBIENTE (FORÇA BRUTA) ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ROC_ALLOC_CONF"] = "max_split_size_mb:32"  # Reduzi ainda mais para 32
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

# FORÇAR O TORCH A INICIALIZAR A GPU ANTES DE QUALQUER COISA DO ULTRALYTICS
torch.cuda.init()

from ultralytics import YOLO


def treinar_modelo():
    # 1. Limpeza total de memória antes de carregar o modelo
    torch.cuda.empty_cache()

    caminho_data_yaml = r"D:\ArgosCam\scripts\argos_gate.yaml"
    diretorio_runs = r"D:\ArgosCam\Runs"

    # 2. Carregar modelo puramente na CPU primeiro
    model = YOLO("yolo11n.pt")

    # 3. TREINO COM REQUISITOS MÍNIMOS DE SISTEMA
    model.train(
        data=caminho_data_yaml,
        epochs=80,
        imgsz=640,
        batch=1,  # Forçar batch 1 para isolar se é falta de memória
        device=0,
        workers=0,
        project=diretorio_runs,
        name='ArgosGate_D_Drive',
        exist_ok=True,

        # --- ESTABILIDADE TOTAL ---
        amp=False,
        cache=False,
        mosaic=0.0,  # Desativado
        mixup=0.0,  # Desativado
        fraction=0.1,  # <--- TENTE USAR APENAS 10% DO DATASET PARA TESTAR A "LARGADA"

        # Parâmetros de Rigor
        cls=2.5,
        box=10.0,
        plots=True
    )


if __name__ == "__main__":
    treinar_modelo()
