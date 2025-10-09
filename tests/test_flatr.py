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
