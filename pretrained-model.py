import os
import torch
import sys
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer


# Model settings
model_name = "meta-llama/Llama-2-7b-hf"
download_dir = "pretrained_models"


# Create directory if not exists
if not os.path.exists(download_dir):
    os.makedirs(download_dir)


# Download model weights
def download_model(token):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=token, 
            torch_dtype=torch.float16
        )
        model.save_pretrained(os.path.join(download_dir, model_name), weights_only=True)
        
        # Get model weights URL
        weights_url = model.config._get_weights_url()
        
        # Download pytorch_model.bin
        response = requests.get(weights_url, stream=True)
        with open(os.path.join(download_dir, model_name, "pytorch_model.bin"), "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        
        print(f"Model weights saved to {os.path.join(download_dir, model_name)}")
        print(f"pytorch_model.bin saved to {os.path.join(download_dir, model_name)}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")


# Download tokenizer files
def download_tokenizer(token):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        tokenizer.save_pretrained(os.path.join(download_dir, model_name))
        
        print(f"Tokenizer files saved to {os.path.join(download_dir, model_name)}")
    except Exception as e:
        print(f"Error downloading tokenizer: {str(e)}")


# Load token from environment variable or input
for arg in sys.argv[1:]:
    if arg.startswith('HUGGINGFACE_HUB_TOKEN='):
        token = arg.split('=')[1]
        break


# Download model and tokenizer
download_model(token)
download_tokenizer(token)


# Verify model weights and tokenizer
model_path = os.path.join(download_dir, model_name)
print(f"Model path: {model_path}")


if os.path.exists(model_path):
    print("Model directory found")
    weight_files = os.listdir(model_path)
    print(f"Weight files: {weight_files}")
    
    if 'pytorch_model.bin' in weight_files:
        print("pytorch_model.bin found")
    else:
        print("pytorch_model.bin not found")
        
    tokenizer_files = ['tokenizer.json', 'vocab.json', 'merges.txt']
    for file in tokenizer_files:
        if file in weight_files:
            print(f"{file} found")
        else:
            print(f"{file} not found")
else:
    print("Model directory not found")


try:
    # Load model to test
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")