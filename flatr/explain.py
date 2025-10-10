"""Explains Codebase from Flattened Markdown"""
import os
import sys
import flatr
from openai import OpenAI


def main(repo_url: str):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
        )

    md_name = f"{repo_url.rstrip("/").split("/")[-1].removesuffix(".git")}.md"
    flatr.main(repo_url, md_name)
    
    with open(md_name, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    main_instruction = (
        "You are an assistant who answers questions based *only* on the "
        "provided markdown text. If the answer is not in the text, state "
        "that the information is not available in the context. "
        "Maintain the conversation and answer the user's questions. "
        "The markdown text contains all of the information about a "
        "specific GitHub repository, whose name is at the top of the text."
    )
    system_instruction = f"{main_instruction}\n\n[CONTEXT START]\n{md_content}\n[CONTEXT END]\n\n"

    messages = [{'role': 'system', 'content': system_instruction}]

    print("-" * 50)
    print(f"--- Chatbot Initialized with '{md_name}' ---")
    print("You can now ask questions about this repository. Type 'exit' to quit.")
    print("-" * 50)

    while True:
        user_input = input("> ")

        if user_input.lower() == 'exit':
            print("\nEnding chat session. Goodbye!")
            break
        
        messages.append({'role': 'user', 'content': user_input})

        try:
            completion = client.chat.completions.create(
                model="deepseek/deepseek-r1-0528:free",
                messages=messages
            )

            response = completion.choices[0].message.content
            print(response)
            print("-" * 50)

            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(e)
            messages.pop()


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        print("Usage: python -m flatr.explain <github_repo_url>")
        sys.exit(1)

    main(sys.argv[1])
