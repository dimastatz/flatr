"""Explains Codebase from Flattened Markdown"""
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
import bitsandbytes as bnb


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        print("Usage: python -m flatr.explain <github_repo_url>")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    login(token=os.getenv("HF_TOKEN"))
    print("Logged in to Hugging Face")

    # Download Mistral7B from Hugging Face
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device, low_cpu_mem_usage=True
    )

    # Save the quantized model and tokenizer locally
    model.save_pretrained("./quantized_model")
    tokenizer.save_pretrained("./quantized_model")

    # Load the saved quantized model and tokenizer
    model_dir = "./quantized_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, load_in_8bit=True, device_map="auto"
    )

    def generate_text(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    prompt = """What is Python"""

    generated_text = generate_text(prompt)
    print("Generated Text:", generated_text)
