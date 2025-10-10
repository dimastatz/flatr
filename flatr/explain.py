"""Explains Codebase from Flattened Markdown"""
import os
import sys
from google import genai
from google.genai import types
import flatr


def main(repo_url: str):
    """The main loop, handles interaction with an LLM model"""
    # Checks if the environment variable is set
    if os.getenv("GEMINI_API_KEY") is None:
        print("GEMINI_API_KEY environment variable not found.")
        return

    client = genai.Client()

    # Flatten the requested repo
    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    md_name = f"{repo_name}.md"
    flatr.main(repo_url, md_name)

    with open(md_name, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Set a main system instruction, lets the model understand the context
    main_instruction = (
        "You are an assistant who answers questions based *only* on the "
        "provided markdown text. If the answer is not in the text, state "
        "that the information is not available in the context. "
        "Maintain the conversation and answer the user's questions. "
        "The markdown text contains all of the information about a "
        "specific GitHub repository, whose name is at the top of the text."
    )
    system_instruction = (
        f"{main_instruction}\n\n[CONTEXT START]\n{md_content}\n[CONTEXT END]\n\n"
    )

    # Make the instructions as a config for the client
    config = types.GenerateContentConfig(system_instruction=system_instruction)

    print("-" * 50)
    print(f"--- Chatbot Initialized with '{md_name}' ---")
    print("You can now ask questions about this repository. Type 'exit' to quit.")
    print("-" * 50)

    # Loop for the user's questions
    messages = []
    while True:
        user_input = input("> ")

        if user_input.lower() == "exit":
            print("\nEnding chat session. Goodbye!")
            break

        messages.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", config=config, contents=messages
        )

        print(response.text)
        print("-" * 50)

        messages.append(
            types.Content(role="model", parts=[types.Part(text=response.text)])
        )


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        print("Usage: python -m flatr.explain <github_repo_url>")
        sys.exit(1)

    main(sys.argv[1])
