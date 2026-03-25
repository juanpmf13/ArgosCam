import torch
import torch_directml

# Verifica se o DirectML está disponível
device = torch_directml.device()
print(f"Dispositivo: {device}")

# Teste de tensor na GPU AMD
x = torch.tensor([1.0, 2.0]).to(device)
print(f"Tensor na GPU: {x}")