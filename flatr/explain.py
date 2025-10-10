"""Explains Codebase from Flattened Markdown"""
import os
import sys


if __name__ == "__main__":  # pragma: no cover
    print("flatr Explain \n")

    if len(sys.argv) != 2:
        print("Usage: python -m flatr.explain <github_repo_url>")
        sys.exit(1)

    model_path = os.getenv("LLM_MODEL_PATH")
    if not model_path:
        print("Error: LLM_MODEL_PATH environment variable not set")
        print("Download a GGUF model from https://huggingface.co/TheBloke")
        sys.exit(1)

    print(f"model={model_path}")
