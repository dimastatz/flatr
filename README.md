<div align="center">
<h1 align="center"> Flatr </h1> 
<h3>Flatten GitHub Repos into Markdown for LLM-Friendly Code Exploration</br></h3>
<img src="https://img.shields.io/badge/Progress-10%25-red"> <img src="https://img.shields.io/badge/Feedback-Welcome-green">
</br>
</br>
<img src="https://github.com/dimastatz/flatr/blob/main/docs/flatr_logo.png?raw=true" width="256px"> 
</div>

# 📦 Flatr [WIP!]()

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

```python
from flatr import flatr

# Create a flat Markdown from a GitHub repo
repo_url = "https://github.com/user/example-repo"
flattener = flatr(repo_url)
flattener.generate_md("output.md")
```

This generates a **self-contained Markdown file** with all code, docs, and structure from the repo.

---

## Example Output

````markdown
# Repository: ExampleRepo

## Folder: utils

### File: helpers.py
```python
def helper_function(x):
    return x * 2
````

### File: validators.py

```python
def validate(input):
    return input is not None
```

## Folder: main

### File: app.py

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




