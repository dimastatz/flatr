"""Explains Codebase from Flattened Markdown"""
import os
import sys


def __main__():  # pragma: no cover
    if len(sys.argv) != 2:
        print("Usage: python -m flatr.explain <github_repo_url>")
        sys.exit(1)

    model_path = os.getenv("LLM_MODEL_PATH")
    if not model_path:
        print("Error: LLM_MODEL_PATH environment variable not set")
        print("Download a GGUF model from https://huggingface.co/TheBloke")
        sys.exit(1)

    # main(sys.argv[1], model_path)
