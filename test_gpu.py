import torch
import torch_directml

print("--- DIAGNOSTYKA DirectML (AMD na WSL) ---")
print(f"Czy DirectML jest dostępny? : {torch_directml.is_available()}")

if torch_directml.is_available():
    dml_device = torch_directml.device()
    print(f"Urządzenie przypisane: {dml_device}")
    
    # Tworzymy tensor testowy i wrzucamy go na Radeona!
    tensor = torch.tensor([1.0, 2.0]).to(dml_device)
    print(f"Sukces! Tensor wyliczony na karcie: {tensor}")
