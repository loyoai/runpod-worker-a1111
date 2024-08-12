# #!/usr/bin/env bash

# echo "Worker Initiated"

# echo "Symlinking files from Network Volume"
# rm -rf /workspace && \
#   ln -s /runpod-volume /workspace

# if [ -f "/workspace/venv/bin/activate" ]; then
#     echo "Starting WebUI API"
#     source /workspace/venv/bin/activate
#     TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
#     export LD_PRELOAD="${TCMALLOC}"
#     export PYTHONUNBUFFERED=true
#     export HF_HOME="/workspace"
#     python3 /workspace/stable-diffusion-webui/webui.py \
#       --xformers \
#       --no-half-vae \
#       --skip-python-version-check \
#       --skip-torch-cuda-test \
#       --skip-install \
#       --lowram \
#       --opt-sdp-attention \
#       --disable-safe-unpickle \
#       --port 3000 \
#       --api \
#       --nowebui \
#       --skip-version-check \
#       --no-hashing \
#       --no-download-sd-model > /workspace/logs/webui.log 2>&1 &
#     deactivate
# else
#     echo "ERROR: The Python Virtual Environment (/workspace/venv/bin/activate) could not be activated"
#     echo "1. Ensure that you have followed the instructions at: https://github.com/ashleykleynhans/runpod-worker-a1111/blob/main/docs/installing.md"
#     echo "2. Ensure that you have used the Pytorch image for the installation and NOT a Stable Diffusion image."
#     echo "3. Ensure that you have attached your Network Volume to your endpoint."
#     echo "4. Ensure that you didn't assign any other invalid regions to your endpoint."
# fi

# echo "Starting RunPod Handler"
# python3 -u /rp_handler.py





#!/usr/bin/env bash

echo "Worker Initiated"

echo "Symlinking files from Network Volume"
rm -rf /workspace && \
  ln -s /runpod-volume /workspace

if [ -f "/workspace/venv/bin/activate" ]; then
    echo "Starting WebUI API"
    source /workspace/venv/bin/activate
    
    # Preload ip-adapter faceid, insightface, etc.
    echo "Preloading ControlNet models"
    python3 - <<EOF
import torch
from pathlib import Path

# Set paths to your models
model_paths = [
    "/workspace/stable-diffusion-webui/models/ControlNet/ip-adapter-faceid-plusv2_sdxl.bin",
    "/workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/insightface/models/buffalo_l/1k3d68.onnx",
    "/workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/insightface/models/buffalo_l/2d106det.onnx",
    "/workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/insightface/models/buffalo_l/det_10g.onnx",
    "/workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/insightface/models/buffalo_l/genderage.onnx",
    "/workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads/insightface/models/buffalo_l/w600k_r50.onnx"
]

# Load models to cache them
for model_path in model_paths:
    if Path(model_path).exists():
        print(f"Loading model: {model_path}")
        torch.load(model_path, map_location=torch.device('cuda'))  # or appropriate loading method
    else:
        print(f"Model not found: {model_path}")

EOF

    TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
    export LD_PRELOAD="${TCMALLOC}"
    export PYTHONUNBUFFERED=true
    export HF_HOME="/workspace"
    python3 /workspace/stable-diffusion-webui/webui.py \
      --xformers \
      --no-half-vae \
      --skip-python-version-check \
      --skip-torch-cuda-test \
      --skip-install \
      --lowram \
      --opt-sdp-attention \
      --disable-safe-unpickle \
      --port 3000 \
      --api \
      --nowebui \
      --skip-version-check \
      --no-hashing \
      --no-download-sd-model > /workspace/logs/webui.log 2>&1 &
    deactivate
else
    echo "ERROR: The Python Virtual Environment (/workspace/venv/bin/activate) could not be activated"
    echo "1. Ensure that you have followed the instructions at: https://github.com/ashleykleynhans/runpod-worker-a1111/blob/main/docs/installing.md"
    echo "2. Ensure that you have used the Pytorch image for the installation and NOT a Stable Diffusion image."
    echo "3. Ensure that you have attached your Network Volume to your endpoint."
    echo "4. Ensure that you didn't assign any other invalid regions to your endpoint."
fi

echo "Starting RunPod Handler"
python3 -u /rp_handler.py
