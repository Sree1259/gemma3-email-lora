import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "google/gemma-3-270m-it"
ADAPTER_DIR = "gemma3-270m-email-lora-adapter"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading base model to MPS...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
).to("mps")

print(f"Loading LoRA adapters from {ADAPTER_DIR}...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

prompt = "Rewrite professionally: Send me the report today. You missed it yesterday."
messages = [{"role":"user","content":prompt}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("mps")

print("\n--- TEST: BEFORE VS AFTER ---")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
    )
print("Input:", prompt)

# Decode only the newly generated tokens
input_length = inputs["input_ids"].shape[1]
generated_tokens = out[0][input_length:]
output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("Output:", output_text)
