""" Flattens github repo """
import os
import shutil
import tempfile
import requests as r


def download(url: str) -> str:
    """Downloads GitHub repo as zip and extracts it
    example url: https://github.com/dimastatz/whisper-flow.git
    downloaded artifact is: https://github.com/dimastatz/whisper-flow/archive/refs/heads/main.zip
    """
    repo_name = url.split("/")[-1].removesuffix(".git")
    url = url.removesuffix(".git") + "/archive/refs/heads/main.zip"
    zip_path = os.path.join(tempfile.mkdtemp(), f"{repo_name}.zip")

    with r.get(url, timeout=120) as req:
        with open(zip_path, "wb") as file:
            file.write(req.raise_for_status() or req.content)

    return zip_path


def cleanup(file_path: str) -> None:
    """Removes file and its parent directory"""
    parent = os.path.dirname(file_path)
    shutil.rmtree(parent)


def execute(url: str) -> str:
    """Main functions"""
    return url
