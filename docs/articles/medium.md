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

Several other tools have attempted to solve similar problems. **Coderoller** focuses on summarizing repositories for AI ingestion, while **flatten-codebase** converts project structures into Markdown or JSON primarily for embedding workflows.

Flatr takes a slightly different path — it prioritizes human readability while maintaining compatibility with LLMs. The result is an artifact that is both machine-consumable and pleasant to read.

## Looking Ahead

There’s room to expand Flatr further. Future directions include selective flattening that respects file patterns, semantic grouping of related code, integration with vector databases for embedding-based search, and direct connections to LLM APIs for automated analysis.

The long-term vision is simple: make complex software repositories easier to reason about — for both humans and machines.

## Final Thoughts

As AI becomes a standard tool in software development, the way we organize and feed information to models will define how effective they can be. Flatr aims to make that process easier by turning complexity into clarity.

You can explore the project on GitHub: [github.com/dimastatz/flatr](https://github.com/dimastatz/flatr).

If you work with LLMs, embeddings, or automated code analysis, give Flatr a try. Sometimes, the best way to understand something complex is to make it flat.
