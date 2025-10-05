""" Flattens github repo """
import os
import sys
import shutil
import tempfile
import zipfile
import typing
import requests as r


def download(repo_url: str) -> str:
    """Downloads GitHub repo as zip and extracts it
    example url: https://github.com/dimastatz/whisper-flow.git
    downloaded artifact is: https://github.com/dimastatz/whisper-flow/archive/refs/heads/main.zip
    """
    repo_name = repo_url.split("/")[-1].removesuffix(".git")
    repo_url = repo_url.removesuffix(".git") + "/archive/refs/heads/main.zip"
    zip_path = os.path.join(tempfile.mkdtemp(), f"{repo_name}.zip")

    with r.get(repo_url, timeout=120) as req:
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
        ".js",
        ".java",
        ".cpp",
        ".h",
        ".cs",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".ts",
        ".html",
        ".css",
    }
    readme_patterns = {"readme", "README"}

    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1] in code_extensions or any(
                p in file.lower() for p in readme_patterns
            ):
                results.append(os.path.join(root, file))

    return sorted(results)


def get_dir_structure_tree(directory: str) -> typing.List[str]:
    """Returns a directory structure tree"""
    tree_lines = []
    first_iteration = 2 # Flag for skipping the first and second iteration of the loop (avoid having the top level folder name in the tree)
    for root, dirs, files in os.walk(directory):
        # Skipping the first and second iteration
        if first_iteration != 0:
            first_iteration -= 1
            continue
        
        # Sort dirs and files for consistent order
        dirs.sort()
        files.sort()

        rel_path = os.path.relpath(root, directory)
        if rel_path == ".":
            level = 0
        else:
            level = rel_path.count(os.sep) - 1 #Reducing 1 to avoid having an extra indentation, compensating for the top level folder skip
            indent = " " * 4 * level
            tree_lines.append(f"{indent}- {os.path.basename(root)}")

        subindent = " " * 4 * (level + 1)
        for f in files:
            tree_lines.append(f"{subindent}- {f}")

    return tree_lines


def write_markdown(files: list, tree: list, title: str, output_path: str) -> None:
    """Writes file contents to markdown file"""
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"\n# Repo: {title}\n")

        out.write(f"\n## Directory Structure:\n")
        out.write("```text\n")
        out.write("\n".join(tree))
        out.write("\n```\n\n")

        for file_path in files:
            # Get relative path for header
            filename = os.path.basename(file_path)

            # Write file header
            out.write(f"\n## File: {filename}\n\n")

            # Write file contents in code block
            ext = os.path.splitext(filename)[1][1:] or "text"
            out.write(f"````{ext}\n")

            with open(file_path, "r", encoding="utf-8") as file:
                out.write(file.read())

            out.write("\n````\n")


def main(repo_url: str, repo_name: str, output_md: str):  # pragma: no cover
    """Run all flow"""
    print(f"Downloading {repo_url} ...")
    zip_path = download(repo_url)
    print(f"Extracting {zip_path} ...")
    extract_dir = unzip(zip_path)
    print(f"Finding files in {extract_dir} ...")
    files = find_files(extract_dir)
    tree = get_dir_structure_tree(extract_dir)
    print(f"Writing markdown to {output_md} ...")
    write_markdown(files, tree, repo_name, output_md)
    print("Cleaning up ...")
    cleanup(zip_path)
    print(f"Done! Markdown file created: {output_md}")


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        print("Usage: python -m flatr <github_repo_url>")
        sys.exit(1)
    url = sys.argv[1]
    name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    md = f"{name}.md"

    main(url, name, md)
