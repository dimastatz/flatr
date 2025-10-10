
# Repo: https://github.com/dimastatz/flatr


## File: flatr-main/.github/workflows/docker-image.yml

````yml
name: Docker Image CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:

  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build the Docker Test image
      run: docker build --tag flattr-image --build-arg CACHEBUST=$(date +%s) . --file Dockerfile.test

````

## File: flatr-main/.vscode/settings.json

````json
{
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
````

## File: flatr-main/README.md

````md
<div align="center">
<h1 align="center"> Flatr </h1> 
<h3>Flatten GitHub Repos into Markdown for LLM-Friendly Code Exploration</br></h3>
<img src="https://img.shields.io/badge/Progress-80%25-red"> <img src="https://img.shields.io/badge/Feedback-Welcome-green">
</br>
</br>
<img src="https://github.com/dimastatz/flatr/blob/main/docs/flatr_logo.png?raw=true" width="256px"> 
</div>

# 📦 Flatr

**flatr** is a Python library that takes any GitHub repository and creates a **flat Markdown (`.md`) file** containing the entire codebase and documentation. It is designed to make codebases **easier to feed into LLMs** for tasks like code explanation, summarization, and interactive documentation.

---

## 🎯 Problem Scope

Modern software projects are often **spread across multiple directories and files**, making it difficult for both humans and AI models to comprehend the codebase efficiently. Large Language Models (LLMs) face these challenges:

1. **Context Window Limitations** – LLMs can only process a limited amount of text at a time. Hierarchical repositories with many files make it hard for models to reason about the entire project.
2. **Scattered Documentation** – README files and docstrings are often separate from code, creating gaps in understanding.
3. **Navigation Complexity** – Humans also spend time jumping between folders and files to understand code dependencies.

**Why Markdown is Better for LLMs:**

* **Flat Structure:** All code and documentation are in a single file, making it easier for the model to process.
* **Preserved Hierarchy via Headers:** Markdown headers (`#`, `##`, `###`) retain the logical organization of folders and files without breaking the flat flow.
* **Syntax Awareness:** Fenced code blocks (` ```python `) preserve language context, helping LLMs understand code semantics.
* **Human and Machine Readable:** Markdown is easy to read for developers and can be ingested directly by AI models.

By converting a repository into a **flattened Markdown**, flatr ensures that the **entire project is accessible in one coherent view**, maximizing the usefulness of LLMs and interactive tools.

---

## ⚡ Features

* Fetch any public GitHub repository by URL.
* Flatten repository structure into a single Markdown file.
* Preserve folder and file hierarchy using Markdown headers.
* Wrap code in fenced code blocks with syntax highlighting.
* Include README and inline documentation.
* Optional metadata: file size, lines of code, last commit info.

---

## 🚀 Installation

```bash
pip install flatr
```

---

## 💻 Usage

```bash
# Create a flat Markdown from a GitHub repo
repo_url = "https://github.com/dimastatz/flatr"
python -m flatr.flatr repo_url 
```

This generates a **self-contained Markdown file** with all code, docs, and structure from the repo.

---

## Example Output


### Repository: ExampleRepo

#### File: utils/helpers.py
```python
def helper_function(x):
    return x * 2
```

#### File: validators.py
```python
def validate(input):
    return input is not None
```

#### File: main/app.py
```python
from utils.helpers import helper_function
```

---

## 🔮 Future Applications

flatr can be used to build **interactive applications and developer tools**, including:

- **Interactive README files** – Ask questions about your code or get explanations directly inside the documentation.  
- **“Chat to Code” applications** – Use LLMs to navigate, analyze, and reason about your codebase.  
- **Fast navigation of large codebases** – Quickly jump between functions, classes, and modules in a single Markdown file.  
- **Knowledge base integration** – Ingest repositories into RAG pipelines for semantic search and documentation.  
- **Automated code analysis** – Summarize, refactor, or detect issues using AI models.

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests for new features, bug fixes, or multi-language support.

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.





````

## File: flatr-main/docs/articles/medium.md

````md
# A Simple Way to Explore Codebases with LLMs


## The Problem: When LLMs Meet Real Codebases

Large language models are becoming powerful tools for developers — they can summarize, refactor, and even reason about software. Yet, they face a fundamental limitation: code lives in trees, while LLMs read in lines.

A modern repository can contain hundreds of files across different folders, languages, and documentation formats. Feeding such a structure into an AI model quickly runs into three practical challenges. First, the limited context window of most LLMs restricts how much content can be processed at once. Second, important context is scattered across code, comments, and documentation. Third, even for humans, understanding a repository often means jumping between files and piecing together mental connections.

The result is incomplete understanding and weak reasoning — both for AI and for developers trying to use it.

## Flatr

**Flatr**, a lightweight tool that takes any GitHub repository and flattens it into a single Markdown document. The idea is simple: instead of navigating dozens of files, you get one continuous document that retains the folder hierarchy and includes all the code and documentation in order.

Each file’s content is wrapped in a readable, structured format, so both humans and LLMs can move through it without losing context. Optional metadata such as file size, last commit, and line count helps preserve additional information that might be useful during analysis.

With this approach, Flatr turns a complex, multi-file repository into a single linear artifact that’s easy to process, index, and understand.

## Applications and Use Cases

Once a repository becomes a single document, new possibilities emerge. Developers can feed it into large language models to perform comprehensive code analysis, generate documentation, or even detect bugs. Product teams can build knowledge extraction or semantic search pipelines on top of flattened codebases. AI-driven refactoring and code-review tools can analyze projects without the friction of switching between files.

For organizations that manage multiple repositories, Flatr can be part of an automated documentation or observability pipeline — transforming source code into an accessible knowledge base that’s compatible with modern AI systems.

## The Technology Behind the Tool

Flatr is built in **Python** and follows a modular, minimalistic design philosophy. It uses `requests` for GitHub data retrieval, standard libraries like `os` and `pathlib` for file traversal, and simple Markdown rendering utilities for formatting.

It includes a lightweight command-line interface, Docker support for reproducible builds, and a test suite based on `pytest`. The entire system is designed to be easily integrated into larger workflows, making it a practical addition to modern AI and DevOps pipelines.

## The Ecosystem: Similar Tools

Several other tools have attempted to solve similar problems. Some focuse on summarizing repositories for AI ingestion, other convert project structures into Markdown or JSON primarily for embedding workflows.

Flatr takes a slightly different path — it prioritizes human readability while maintaining compatibility with LLMs. The result is an artifact that is both machine-consumable and pleasant to read.

## Looking Ahead

There’s room to expand Flatr further. Future directions include selective flattening that respects file patterns, semantic grouping of related code, integration with vector databases for embedding-based search, and direct connections to LLM APIs for automated analysis.

The long-term vision is simple: make complex software repositories easier to reason about — for both humans and machines.

## Final Thoughts

As AI becomes a standard tool in software development, the way we organize and feed information to models will define how effective they can be. Flatr aims to make that process easier by turning complexity into clarity.

You can explore the project on GitHub: [github.com/dimastatz/flatr](https://github.com/dimastatz/flatr).

If you work with LLMs, embeddings, or automated code analysis, give Flatr a try. Sometimes, the best way to understand something complex is to make it flat.

````

## File: flatr-main/flatr/__init__.py

````py
""" add package version """

__version__ = "1.0.2"

````

## File: flatr-main/flatr/explain.py

````py
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

````

## File: flatr-main/flatr/flatr.py

````py
""" Flattens github repo """
import os
import sys
import shutil
import tempfile
import zipfile
import typing
import re
import requests as r


def download(repo_url: str) -> str:
    """Downloads GitHub repo as zip and extracts it
    example url: https://github.com/dimastatz/whisper-flow.git
    downloaded artifact is: https://github.com/dimastatz/whisper-flow/archive/refs/heads/main.zip
    """
    repo_name = repo_url.split("/")[-1].removesuffix(".git")
    repo_url = repo_url.removesuffix(".git") + "/archive/refs/heads/main.zip"
    zip_path = os.path.join(tempfile.mkdtemp(), f"{repo_name}.zip")

    with r.get(repo_url, timeout=180) as req:
        with open(zip_path, "wb") as file:
            file.write(req.raise_for_status() or req.content)

    return zip_path


def cleanup(file_path: str) -> None:
    """Removes file and its parent directory"""
    parent = os.path.dirname(file_path)
    shutil.rmtree(parent)


def unzip(zip_path: str) -> str:
    """Extracts zip file to a new directory next to it"""
    extract_path = os.path.splitext(zip_path)[0]
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(extract_path)
    return extract_path


def find_files(directory: str) -> typing.List[str]:
    """Find all code and readme files in directory"""
    code_extensions = {
        ".py",
        ".pyx",
        ".pyd",
        ".pyi",  # Python
        ".js",
        ".jsx",
        ".ts",
        ".tsx",  # JavaScript/TypeScript
        ".java",
        ".scala",
        ".kt",
        ".kts",  # JVM
        ".cpp",
        ".hpp",
        ".c",
        ".h",  # C/C++
        ".cs",
        ".vb",
        ".fs",  # .NET
        ".go",
        ".rs",
        ".rb",  # Go/Rust/Ruby
        ".php",
        ".php4",
        ".php5",
        ".phtml",  # PHP
        ".swift",
        ".m",
        ".mm",  # Apple
        ".R",
        ".r",  # R
        ".sql",
        ".mysql",
        ".pgsql",  # SQL
        ".sh",
        ".bash",
        ".zsh",  # Shell
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",  # Web
        ".xml",
        ".yaml",
        ".yml",
        ".json",
        ".toml",  # Config
        ".lua",
        ".pl",
        ".pm",  # Lua/Perl
        ".ex",
        ".exs",  # Elixir
        ".elm",
        ".clj",
        ".ml",  # Functional
        ".dart",
        ".gradle",
        ".groovy",  # Others
        ".f90",
        ".f95",
        ".f03",  # Fortran
        ".asm",
        ".s",  # Assembly
        ".v",
        ".sv",  # Verilog
        ".mat",  # MATLAB
        ".txt",
        ".md",  # Text/Markdown
    }
    # readme_patterns = {"readme"}

    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1] in code_extensions:
                results.append(os.path.join(root, file))

    return sorted(results)


def count_backticks(file_path: str) -> int:
    """
    Checks amount of backticks in the file to not break the output markdown file,
    returns 4 (default backticks amount) if none found or file is safe
    """
    with open(file_path, "r", encoding="utf-8") as file:
        file_contents = file.read()

    backticks_list = (
        re.findall(r"`+", file_contents) or None
    )  # Regular expression to check for one or more backticks
    if backticks_list is None:
        return 4

    longest_occurrence = max(backticks_list, key=len)
    backtick_count = len(longest_occurrence)

    if backtick_count >= 4:
        return backtick_count + 1
    return 4


def write_markdown(base_path: str, files: list, title: str, output_path: str) -> None:
    """Writes file contents to markdown file"""
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"\n# Repo: {title}\n\n")

        for file_path in files:
            # Get relative path for header
            filename = os.path.relpath(file_path, base_path)

            # Write file header
            out.write(f"\n## File: {filename}\n\n")

            # Write file contents in code block
            ext = os.path.splitext(filename)[1][1:] or "text"
            if ext in (
                "txt",
                "md",
            ):  # Check for text or md file in order to not break the output file format
                backticks_amount = count_backticks(file_path)
                out.write(f"{'`'*backticks_amount}{ext}\n")

                with open(file_path, "r", encoding="utf-8") as file:
                    out.write(file.read())

                out.write(f"\n{'`'*backticks_amount}\n")
            else:
                out.write(f"````{ext}\n")
                with open(file_path, "r", encoding="utf-8") as file:
                    out.write(file.read())

                out.write("\n````\n")


def main(repo_url: str, output_md: str):  # pragma: no cover
    """Run all flow"""
    print(f"Downloading {repo_url} ...")
    zip_path = download(repo_url)
    print(f"Extracting {zip_path} ...")
    extract_dir = unzip(zip_path)
    print(f"Finding files in {extract_dir} ...")
    files = find_files(extract_dir)
    print(f"Writing markdown to {output_md} ...")
    write_markdown(extract_dir, files, repo_url, output_md)
    print("Cleaning up ...")
    cleanup(zip_path)
    print(f"Done! Markdown file created: {output_md}")


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python -m flatr <github_repo_url> [output_md_file]")
        sys.exit(1)
    url = sys.argv[1]
    name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    md = sys.argv[1] if len(sys.argv) == 3 else f"{name}.md"

    main(url, md)

````

## File: flatr-main/requirements.txt

````txt
pytest==7.3.2
black==23.3.0
pylint==3.2.3
pytest-cov==4.1.0
requests==2.31.0
````

## File: flatr-main/run.sh

````sh
#!/bin/bash

abort()
{
    echo "*** FAILED ***" >&2
    exit 1
}

if [ "$#" -eq 0 ]; then
    echo "Wrong argument is provided. Usage:
            '-local' to build local environment
            '-test' to run linter, formatter and tests
            '-docker' to build and run docker image"

elif [ $1 = "-local" ]; then
    trap 'abort' 0
    set -e
    echo "Running format, linter and tests"
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r ./requirements.txt

    black flatr tests
    pylint --fail-under=9.9 flatr tests
    pytest --ignore=tests/benchmark --cov-fail-under=95 --cov flatr -v tests
elif [ $1 = "-test" ]; then
    trap 'abort' 0
    set -e
    
    echo "Running format, linter and tests"
    source .venv/bin/activate
    black flatr tests
    pylint --fail-under=9.9 flatr tests
    pytest --ignore=tests/benchmark --cov-fail-under=95 --cov --log-cli-level=INFO flatr -v tests
elif [ $1 = "-docker" ]; then
    echo "Building and running docker image"
    docker stop flatr-container
    docker rm flatr-container
    docker rmi flatr-image
    # build docker
    docker build --tag flatr-image --build-arg CACHEBUST=$(date +%s) . --file Dockerfile.test
elif [ $1 = "-deploy-package" ]; then
    echo "Running WhisperFlow package setup"
    pip install twine
    pip install wheel
    python setup.py sdist bdist_wheel
    rm -rf .venv_test
    python3 -m venv .venv_test
    source .venv_test/bin/activate
    pip install ./dist/flatr-0.2-py3-none-any.whl
    pytest --ignore=tests/benchmark --cov-fail-under=95 --cov whisperflow -v tests
    # twine upload ./dist/*
else
  echo "Wrong argument is provided. Usage:
            '-local' to build local environment
            '-test' to run linter, formatter and tests
            '-docker' to build and run docker image"
fi

trap : 0
echo >&2 '*** DONE ***'
````

## File: flatr-main/setup.py

````py
from pathlib import Path
from setuptools import setup
from flatr import __version__
from pkg_resources import parse_requirements


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='flatr',
    version=__version__,
    url='https://github.com/dimastatz/flatr',
    author='Dima Statz',
    author_email='dima.statz@gmail.com',
    py_modules=['flatr'],
    python_requires=">=3.9",
    install_requires=[
        str(r)
        for r in parse_requirements(
            Path(__file__).with_name("requirements.txt").open()
        )
    ],
    description='Flatten GitHub Repos into Markdown for LLM-Friendly Code Exploration',
    long_description = long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={'': ['static/*']},
)
````

## File: flatr-main/tests/__init__.py

````py

````

## File: flatr-main/tests/test_flatr.py

````py
""" Tests for flattr module """
import os
import requests
import flatr.flatr


def test_execute():
    """Tests execute function"""
    repo_url = "https://github.com/dimastatz/flatr.git"
    zip_file = flatr.flatr.download(repo_url)
    assert zip_file.endswith(".zip")

    extract_path = flatr.flatr.unzip(zip_file)
    assert os.path.exists(extract_path)

    files = flatr.flatr.find_files(extract_path)
    assert len(files) > 0
    assert any(f.endswith(".py") for f in files)

    markdown_path = os.path.join(extract_path, "output.md")
    flatr.flatr.write_markdown(extract_path, files, repo_url, markdown_path)
    assert os.path.exists(markdown_path)

    # Cleanup
    flatr.flatr.cleanup(zip_file)
    assert os.path.exists(zip_file) is False
    assert os.path.exists(extract_path) is False


def test_multiple_md():
    """Validates if all md files written to output file"""
    repo_url = "https://github.com/dimastatz/flatr"

    # Execute the main function
    zip_file = flatr.flatr.download(repo_url)
    extract_path = flatr.flatr.unzip(zip_file)
    files = flatr.flatr.find_files(extract_path)
    markdown_path = os.path.join(extract_path, "output.md")
    flatr.flatr.write_markdown(extract_path, files, repo_url, markdown_path)

    # Get all content of output file to string
    with open(markdown_path, "r", encoding="utf-8") as f:
        output_file_content = f.read().replace(
            "\\", "/"
        )  # Added the replace function for testing on windows

    # Get a list of all md files from GitHub API
    url_parts = repo_url.rstrip("/").split("/")
    owner = url_parts[3]
    repo = url_parts[4]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    all_files = requests.get(api_url, timeout=10).json()["tree"]

    md_files = []
    for file in all_files:
        if file["type"] == "blob" and file["path"].endswith(".md"):
            md_files.append(file["path"])

    # Check if all md files exist in output
    missing_files = []
    for file in md_files:
        if file in output_file_content:
            print(file)
        else:
            missing_files.append(file)

    assert not missing_files

    # Cleanup
    flatr.flatr.cleanup(zip_file)

````
