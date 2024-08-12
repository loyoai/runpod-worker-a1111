import os
import sys
import torch

# Add the stable-diffusion-webui directory to the Python path
sd_webui_path = "/stable-diffusion-webui"
sys.path.append(sd_webui_path)

from modules import shared, devices, sd_models
from modules.paths import models_path

def initialize_sd_model():
    print("Initializing Stable Diffusion model...")
    sd_models.setup_model()
    sd_models.load_model()

def initialize_ipadapter():
    print("Initializing IPAdapter...")
    from extensions.ip-adapter.ipadapter.ip_adapter import IPAdapterPlus
    
    # Initialize IPAdapterPlus
    ip_model = IPAdapterPlus(
        sd_model=shared.sd_model,
        device=devices.device,
        num_tokens=16,
        cross_attention_dim=2048,
        model_path="/stable-diffusion-webui/extensions/ip-adapter/models/ip-adapter-plus-face_sd15.bin"
    )
    
    # Run a dummy inference to cache computations
    dummy_image = torch.randn(1, 3, 224, 224).to(devices.device)
    ip_model.get_image_embeds(dummy_image)

def main():
    os.makedirs(models_path, exist_ok=True)
    
    devices.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    initialize_sd_model()
    initialize_ipadapter()
    
    print("Caching complete. Models are initialized and ready for faster startup.")

if __name__ == "__main__":
    main()
