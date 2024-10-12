import transformers
import os


# Set token from environment variable
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")


# Login to Hugging Face Hub
transformers.huggingface_hub.login(token)


# Rest of your code...
model_name = "meta-llama/Llama-2-7b-hf"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)


# Save model locally (optional)
model.save_pretrained(f"pretrained_models/{model_name}")