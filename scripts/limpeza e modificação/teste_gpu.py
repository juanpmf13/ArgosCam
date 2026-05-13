import os
import torch

# PASSO CRUCIAL: Sem isso, a RX 6750 XT causa o erro 0xC0000005
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"


def verificar():
    print("--- Verificação ArgosCam GPU ---")
    cuda_disponivel = torch.cuda.is_available()
    print(f"PyTorch reconhece GPU AMD? {cuda_disponivel}")

    if cuda_disponivel:
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
        # Teste de memória: cria um tensor e joga na placa
        try:
            x = torch.ones(1).cuda()
            print("✅ Comunicação realizada com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao mover tensor para GPU: {e}")
    else:
        print("❌ GPU não detectada. Verifique se o HIP SDK está no PATH.")


if __name__ == "__main__":
    verificar()