<div align="center">
<h1 align="center"> Flattr </h1> 
<h3>Flattr: The LLM Context GeneratorFlattr</br></h3>
<img src="https://img.shields.io/badge/Progress-10%25-red"> <img src="https://img.shields.io/badge/Feedback-Welcome-green">
</br>
</br>
<kbd>
<img src="https://github.com/dimastatz/flattr/blob/main/flattr_logo.png?raw=true" width="256px"> 
</kbd>
</div>

## What is Flattr
Flattr: is a lightweight, open-source Python library designed to solve the problem of feeding large, multi-file codebases to Large Language Models (LLMs). It converts any Git repository or local folder into a single, clean, highly structured Markdown file, optimized for the LLM's context window.Stop manually copying files and wasting tokens on logs and binaries—give your LLM the perfect context every time.✨ FeaturesRepo Ingestion: Accepts both GitHub URLs (clones automatically) and local file paths.Intelligent Filtering: Automatically parses .gitignore and ignores common non-source files (logs, images, binaries, dependency folders, etc.) to ensure Token Efficiency.Structured Output: Generates a single .md file with clear Markdown headers to delineate directories and file boundaries, providing essential Semantic Understanding.LLM-Ready Format: Uses code fences (```language) to clearly separate code, making it easy for the model to parse syntax and file locations.🚀 InstallationFlattr is a Python package. We recommend installing it within a virtual environment.# 1. Clone the Flattr repository (for development/latest version)
git clone https://github.com/your-username/flattr.git
cd flattr

## How To
```bash
pip install flattr
```

### OR for local development:
# pip install -r requirements.txt
💡 UsageThe primary function of Flattr is simple: generate the context file.Command Line Interface (CLI)Use the flattr command followed by the repository URL or local path.Example 1: Generating context from a GitHub URLflattr https://github.com/owner/repository-name.git --output codebase.md
Example 2: Generating context from a local directoryflattr ./my-local-project/ --output local_context.md
Python Library UsageYou can also import and use the core function within your own Python scripts:import flattr

## Generate a context file from a URL
flattr.generate_context(
    source="https://github.com/owner/repository-name.git",
    output_file="flattr_context.md",
    # Optional: include specific extensions to override filtering
    include_extensions=[".py", ".ts"] 
)

print("Context file generated successfully!")
⚙️ Output Structure (What the LLM Sees)The generated Markdown file provides a clean roadmap for the LLM:# Project Name: MyAwesomeProject
## Repository Summary
* Source: https://github.com/owner/repository-name.git
* Total Files Parsed: 12

# FILE: /src/core/api_service.py
```python
import requests
# Code for API service here...
FILE: /src/ui/button.tsx// React component code for Button
const Button = () => { /* ... */ };

## 🌟 Future Vision

The true power of Flattr lies in its potential to enable advanced LLM-powered developer tools:

* **"Talk to Code" Interactive Shell:** By using the generated flat file as a system prompt, developers can initiate a long-running, deep-context chat session where the LLM acts as an **Architectural Advisor** for the entire codebase.
* **CI/CD Documentation:** Integration into pipelines to automatically update `README.md` files or generate architectural summaries based on the most recent flattened codebase.
* **Advanced Code Analysis:** Leverage the structured context for automated **Code Style Enforcement** and natural language **Dependency Mapping**.

---
*License: MIT | Contribution guidelines and issue reporting are welcome.*
