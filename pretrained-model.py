import os
import torch
import sys
from transformers import AutoModelForCausalLM


# Model settings
model_name = "meta-llama/Llama-2-7b-hf"
download_dir = "pretrained_models"


# Create directory if not exists
if not os.path.exists(download_dir):
    os.makedirs(download_dir)


# Download model weights
def download_model(token):
    try:
        # Set torch dtype to reduce memory requirements
        torch_dtype = torch.float16
        
        # Use revision to download specific model version
        revision = "main"
        
        # Download and save model
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
if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
    print("Model weights downloaded successfully")
    try:
        # Load model to test
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
else:
    print("Model weights not found")