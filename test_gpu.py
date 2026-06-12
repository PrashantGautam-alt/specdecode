
import torch

def check_gpu():
     print(f"CUDA available: {torch.cuda.is_available()}")

     if torch.cuda.is_available():
         print(f"GPU name: {torch.cuda.get_device_name(0)}")

         total_vram = torch.cuda.get_device_properties(0).total_memory
         total_vram_gb = total_vram/(1024**3)

         print(f"Total_vram: {total_vram_gb:.1f} GB")

if __name__ == "__main__":
    check_gpu()
