import transformers
import sys


# Parse command-line arguments
for arg in sys.argv[1:]:
    if arg.startswith('HUGGINGFACE_HUB_TOKEN='):
        token = arg.split('=')[1]
        break


# Set verbosity to error
transformers.logging.set_verbosity_error()


# Load pre-trained model with authentication token
model_name = "meta-llama/Llama-2-7b-hf"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)


# Save model locally (optional)
model.save_pretrained(f"pretrained_models/{model_name}")