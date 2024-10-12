import os
import torch
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
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
        model.save_pretrained(os.path.join(download_dir, model_name), weights_only=True)
        print(f"Model weights saved to {os.path.join(download_dir, model_name)}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")


# Load token from environment variable or input
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token is None:
    token = input("Enter Hugging Face token: ")


# Download model
download_model(token)


# Verify model weights
model_path = os.path.join(download_dir, model_name)
if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
    print("Model weights downloaded successfully")
else:
    print("Model weights not found")