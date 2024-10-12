from transformers import AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)

model.save_pretrained(f"pretrained_models/{model_name}")