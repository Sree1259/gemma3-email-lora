# Gemma 3 Email LoRA Adapter

**Professional Email Rewriting with LoRA Fine-Tuning**

This repository contains a **fine-tuned LoRA adapter** for the **Gemma 3 270M** language model, designed to rewrite blunt or casual emails into a professional tone. The project demonstrates efficient fine-tuning using **LoRA (Low-Rank Adaptation)**, which allows you to train a small adapter without modifying the full base model, making it lightweight and easy to share (~15MB).

---

## Project Overview

In this project, we:

1. **Set up the environment**

   * Installed PyTorch, Transformers, TRL, PEFT, Datasets, Accelerate, and Hugging Face Hub.
   * Logged in to Hugging Face using an access token to access models and push adapters.

2. **Loaded the base model**

   * Used `google/gemma-3-270m-it` as the base causal language model.

3. **Applied LoRA adapters**

   * Fine-tuned only key parameters (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.) to efficiently adapt the model to professional email rewriting.

4. **Prepared the dataset**

   * Loaded and formatted a custom email dataset (`emails.jsonl`) into instruction-output pairs.
   * Tokenized prompts for training and evaluation.

5. **Fine-tuned the model**

   * Configured training with `SFTTrainer` for 3 epochs.
   * Trained on CPU (or GPU if available) while monitoring evaluation metrics and saving adapter checkpoints.

6. **Saved the adapter**

   * Saved only the LoRA adapter (~15MB) and tokenizer files.
   * Allows any user with the base Gemma 3 model to use the professional email adapter without needing the full 270M model.

7. **Interactive inference**

   * Provided `win_interactive_test.py` for real-time interaction: type a blunt email and get a professional rewrite.

**Outcome:**

* Users can rewrite casual or blunt emails into professional language instantly.
* The adapter is lightweight, portable, and works with the original Gemma 3 270M model.

---

## Installation

Install the required packages:

```bash
pip install -U torch transformers datasets trl accelerate peft huggingface_hub
```

For CPU-only PyTorch (optional):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Login to Hugging Face to access models:

```bash
python -c "from huggingface_hub import login; login(token='YOUR_HF_ACCESS_TOKEN')"
```

---

## Usage

### 1. Fine-Tune 

```bash
python win_interactive_train.py
```

* Trains the LoRA adapter on your custom email dataset.

### 2. Interactive Inference

```bash
python win_interactive_test.py
```

* Type a blunt email and see a professional rewrite.
* Example:

```
[BLUNT EMAIL]: This meeting was a waste of time.
[PROFESSIONAL VERSION]: I appreciate everyone's effort, but I feel the meeting could have been more productive.
```

---

## File Structure

```
gemma3-email-lora/
│
├─ gemma3-270m-email-lora-adapter/   # LoRA adapter files (~15MB)
├─ win_interactive_train.py           # Script for fine-tuning the adapter
├─ win_interactive_test.py            # Script for interactive inference
├─ emails.jsonl                       # Sample email dataset
└─ README.md
```

---

## Notes

* Only the **adapter** and **tokenizer** are saved — not the full base model.
* This approach reduces storage and makes it easier to share fine-tuned models.
* Works with `google/gemma-3-270m-it`.


