"""Functional Gemini-based Q&A over a Markdown file."""
import os
import sys
from typing import List, Dict, Any
import google.generativeai as genai


def read_md(md_path: str) -> str:
    """Read and return the content of a Markdown file."""
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


def build_system_instruction(md_content: str) -> str:
    """Create a system instruction to ground the LLM's answers in the Markdown."""
    return (
        "You are an assistant that answers questions strictly using the provided "
        "Markdown. If the information is not present, say 'Information not available.' "
        "Do not hallucinate.\n\n"
        f"[CONTEXT START]\n{md_content}\n[CONTEXT END]\n\n"
    )


def call_model(
    system_instruction: str,
    messages: List[Dict[str, Any]],
    model: str = "gemini-2.5-flash-lite",
) -> str:
    """Call the Gemini model and return its response."""
    config = {"system_instruction": system_instruction}
    response = genai.models.generate_content(  # pylint: disable=no-member
        model=model, config=config, contents=messages
    )
    return response.text if hasattr(response, "text") else str(response)


def interactive_loop(md_path: str, api_key: str) -> None:
    """Run an interactive Q&A loop grounded in the Markdown file."""
    genai.configure(api_key=api_key)
    md_content = read_md(md_path)
    system_instruction = build_system_instruction(md_content)

    print(f"Using Markdown file: {md_path}")
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
            answer = call_model(system_instruction, messages)
            print("\n" + answer + "\n" + "-" * 60)
            messages.append({"role": "model", "content": [{"text": answer}]})
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break


def main(argv: List[str]) -> None:
    """Main entry point for the script."""
    if len(argv) != 2:
        print("Usage: python -m flatr.explain <path-to-markdown.md>")
        sys.exit(1)

    md_path = argv[1]
    if not os.path.exists(md_path):
        print(f"Error: Markdown file not found at '{md_path}'")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set. Run: export GEMINI_API_KEY=your_key")
        sys.exit(1)

    interactive_loop(md_path, api_key)


if __name__ == "__main__":
    main(sys.argv)
