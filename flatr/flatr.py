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
