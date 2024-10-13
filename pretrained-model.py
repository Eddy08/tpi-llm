import os
import torch
import sys
from transformers import AutoModelForCausalLM

## Logs for GPU
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.device_count())  # Should print: 1
print(torch.cuda.current_device())  # Should print: 0


# Model settings
model_name = "meta-llama/Llama-2-7b-hf"
download_dir = "pretrained_models"


# Create directory if not exists
if not os.path.exists(download_dir):
    os.makedirs(download_dir)


# Download model weights
def download_model(token):
    try:
        torch_dtype = torch.float16
        revision = "main"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=token, 
            revision=revision, 
            torch_dtype=torch_dtype
        )
        model.save_pretrained(os.path.join(download_dir, model_name), weights_only=True)
        
        print(f"Model weights saved to {os.path.join(download_dir, model_name)}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")


# Load token from environment variable or input
for arg in sys.argv[1:]:
    if arg.startswith('HUGGINGFACE_HUB_TOKEN='):
        token = arg.split('=')[1]
        break


# Download model
download_model(token)


# Verify model weights
model_path = os.path.join(download_dir, model_name)
print(f"Model path: {model_path}")  # Print model path


if os.path.exists(model_path):
    print("Model directory found")
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        print("Model weights found")
    else:
        print("Model weights not found (missing pytorch_model.bin)")
else:
    print("Model directory not found")


try:
    # Load model to test
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")