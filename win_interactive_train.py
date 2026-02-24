import torch
import sys
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def ask_to_proceed(step_name):
    print(f"\n[STEP] {step_name}")
    choice = input("Shall we proceed? (y/n): ").lower()
    if choice != 'y':
        print("Exiting...")
        sys.exit()

def print_explanation(title, text):
    print("\n" + "="*50)
    print(f"🔹 {title.upper()}")
    print("="*50)
    print(text)
    print("="*50)

# 1) Setup
print_explanation("Step 1: The Engine (Hardware)", 
    "🚀 WE ARE CHECKING YOUR MAC'S POWERHOUSE.\n"
    "Visualization: Think of the CPU as a 'math teacher' and the GPU (MPS) as a 'room full of calculators'.\n"
    "We use 'MPS' (Metal Performance Shaders) which lets your Mac's GPU handle the heavy lifting.\n"
    "Without this, training would feel like walking through mud.")

ask_to_proceed("Initialize Hardware")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"✅ Hardware Ready: Using {device.upper()} acceleration.")

# 2) Load Model
print_explanation("Step 2: The Brain (The Model)", 
    "🧠 LOADING THE 'GEMMA 3 270M' BRAIN.\n"
    "Visualization: 270 Million Parameters means the model has 270,000,000 'tuning knobs'.\n"
    "Here is what a tiny piece of that 'brain' looks like (weights):\n"
    "   [[ 0.012, -0.045,  0.089 ],\n"
    "    [-0.033,  0.112, -0.007 ],\n"
    "    [ 0.056, -0.021,  0.044 ]]\n"
    "We load it in 'float32' mode to keep these tiny decimals precise.")

ask_to_proceed("Load Model & Tokenizer")
MODEL_ID = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
).to(device)
print("✅ Model loaded! It's currently a general-purpose 'it' model.")

# 3) LoRA Setup
print_explanation("Step 3: The Post-it Note (LoRA)", 
    "📝 APPLYING LoRA (Low-Rank Adaptation).\n"
    "Visualization: 3.7 Million Parameters is roughly 15MB of data.\n"
    "These are 'Delta Matrices' (the difference we want to teach the model).\n"
    "If the model currently thinks 'Send' is 0.5, our LoRA might say '+0.2',\n"
    "changing it to 0.7 to nudge it toward a professional tone.")

ask_to_proceed("Apply LoRA Adapters")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4) Load Dataset
print_explanation("Step 4: The Schooling (Dataset)", 
    "📚 PREPARING YOUR EMAIL DATASET.\n"
    "Visualization: The model doesn't read letters, it reads NUMBERS (Token IDs).\n"
    "Example Translation:\n"
    "   Text:  'Please send...' \n"
    "   Tokens: [ 2341, 1402 ]\n"
    "We wrap these in a conversation template so the model knows who is talking.")

ask_to_proceed("Load and Format Data")
ds = load_dataset("json", data_files="emails.jsonl", split="train")

def format_prompts(ex):
    messages = [
        {"role": "user", "content": ex["instruction"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

ds = ds.map(format_prompts)

# Show a sample of tokenization
sample_text = ds[0]['text']
sample_tokens = tokenizer.encode(sample_text)[:10]
print(f"\n[DATA PREVIEW]\nRaw Text: {sample_text[:60]}...")
print(f"Token IDs (what the model sees): {sample_tokens} ...")

ds = ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = ds["train"], ds["test"]
print(f"\n📊 Training on {len(train_ds)} manners examples. Testing on {len(eval_ds)} hidden examples.")

# 5) Trainer Config
print_explanation("Step 5: The Study Session (Training)", 
    "⏳ STARTING THE FINE-TUNING LOOP.\n"
    "Visualization: Think of this as an 'exam'. The model tries to guess the answer,\n"
    "looks at the 'Loss' (the error score), and adjusts its Post-it Note to do better next time.\n"
    "We will run 3 Epochs—meaning the model reads the entire manners book 3 times.")

ask_to_proceed("Start Training Loop")
args = SFTConfig(
    output_dir="output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    use_cpu=True,
    bf16=False,
    fp16=False,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=10,
    save_steps=10,
    save_total_limit=2,
    report_to="none",
    max_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=args,
)

trainer.train()

# 6) Save
print_explanation("Step 6: Graduating (Saving)", 
    "🎓 SAVING YOUR TRAINED ADAPTER.\n"
    "Visualization: We don't save the whole library. We only save the Post-it Note!\n"
    "The file is only ~15MB. You can share this tiny file with anyone who has Gemma 3,\n"
    "and they will instantly have your 'Professional Email' version.")

ask_to_proceed("Save the manners-adapter")
ADAPTER_PATH = "gemma3-270m-email-lora-adapter"
trainer.model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print(f"\n🎉 GRADUATION COMPLETE! Adapter saved to {ADAPTER_PATH}")
print("\n👉 To see the model in action, run: python3 interactive_test.py")
