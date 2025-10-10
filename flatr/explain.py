"""Explains Codebase from Flattened Markdown"""
import os
import sys
from ctransformers import AutoModelForCausalLM


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        print("Usage: python -m flatr.explain <github_repo_url>")
        sys.exit(1)

    model_path = os.getenv("LLM_MODEL_PATH")
    if not model_path:
        print("Error: LLM_MODEL_PATH environment variable not set")
        print("Download a GGUF model from https://huggingface.co/TheBloke")
        sys.exit(1)

    print(f"model={model_path}")

    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="llama",
        context_length=2048,
        max_new_tokens=512,
        gpu_layers=0 
    )

    print(llm("What is your model name", temperature=0.7))
