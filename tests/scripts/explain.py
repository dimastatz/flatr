"""Functional Gemini-based Q&A over a Markdown file."""
import os
import sys
import tempfile
from urllib.parse import urlparse

from typing import List, Dict, Any
from google import genai
from google.genai.types import GenerateContentConfig

import flatr.flatr


def read_md(md_path: str) -> str:  # pragma: no cover
    """Read and return the content of a Markdown file."""
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


def interactive_loop(  # pragma: no cover
    repo_url: str, api_key: str, model: str = "gemini-2.5-flash-lite"
) -> None:
    """Run an interactive Q&A loop grounded in the Markdown file."""
    client = genai.Client(api_key=api_key)
    model = client.get_model(model)
    print(model.generate_content("Explain large language "))

    with tempfile.NamedTemporaryFile(delete=True) as temp:
        flatr.flatr.main(repo_url=repo_url, output_md=temp.name)
        md_content = read_md(temp.name)

    config = GenerateContentConfig(
        system_instruction=[
            "You are a Software Engineer that answers questions using the provided Markdown",
            "Markdown contains the source code from a github repo specified in this file."
            "If the information is missing, say 'Information not available.' Do not hallucinate.",
            f"[CONTEXT START]\n{md_content}\n[CONTEXT END]",
        ]
    )

    print("Ask questions about the code (type 'exit' to quit).")

    messages: List[Dict[str, Any]] = []
    while True:
        try:
            question = input("> ").strip()
            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not question:
                continue

            messages.append({"role": "user", "content": [{"text": question}]})
            response = client.models.generate_content(
                model=model, contents=messages, config=config
            )  # Warm up the model

            print("\n" + response.text + "\n" + "-" * 60)
            messages.append({"role": "model", "content": [{"text": response.text}]})
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break


def main(argv: List[str]) -> None:  # pragma: no cover
    """Main entry point for the script."""
    if len(argv) != 2:
        print("Usage: python -m flatr.explain <path-to-markdown.md>")
        sys.exit(1)

    url = urlparse(argv[1])
    if not all([url.scheme in ("http", "https"), url.netloc]):
        print(f"Error: Bad url '{url}'")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set. Run: export GEMINI_API_KEY=your_key")
        sys.exit(1)

    interactive_loop(url.geturl(), api_key)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv)
