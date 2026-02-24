import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

MODEL_ID = "google/gemma-3-270m-it"
ADAPTER_DIR = "gemma3-270m-email-lora-adapter"

# ✅ Proper device detection
device = "cuda" if torch.cuda.is_available() else "cpu"

def print_header(text):
    print("\n" + "="*50)
    print(f"🚀 {text.upper()}")
    print("="*50)

print_header("Initializing Interactive Inference")

print(f"Using device: {device}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)

# ✅ Move model to correct device
model.to(device)

try:
    print(f"Applying your fine-tuned LoRA adapters from '{ADAPTER_DIR}'...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.to(device)  # ensure adapter model also on same device
except Exception as e:
    print(f"\n❌ ERROR: Could not find the trained adapter folder '{ADAPTER_DIR}'.")
    print("Please run 'python interactive_train.py' first to train the model!")
    sys.exit()

model.eval()

print_header("Model Ready!")
print("Type a 'blunt' email and the model will rewrite it professionally.")
print("Type 'quit' or 'exit' to stop.")

while True:
    user_input = input("\n[BLUNT EMAIL]: ").strip()

    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    if not user_input:
        continue

    prompt = f"Rewrite professionally: {user_input}"
    messages = [{"role": "user", "content": prompt}]

    # First build formatted prompt text
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Then tokenize properly (returns a dictionary)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    )

    # Move tensors to correct device
    inputs = {k: v.to(device) for k, v in inputs.items()} 

    print("Generating...")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = out[0][input_length:]
    output_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    print(f"\n[PROFESSIONAL VERSION]:")
    print("-" * 30)
    print(output_text if output_text else "(Model produced empty output.)")
    print("-" * 30)