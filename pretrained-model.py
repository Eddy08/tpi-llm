import transformers
import os


# Set token from environment variable
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")


# Set verbosity to error
transformers.logging.set_verbosity_error()


# Load pre-trained model with authentication token
model_name = "meta-llama/Llama-2-7b-hf"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)


# Save model locally (optional)
model.save_pretrained(f"pretrained_models/{model_name}")