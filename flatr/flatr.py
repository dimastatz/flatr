""" Flattens github repo """
import os
import tempfile

import urllib.parse as p
import requests as r


def download(url: str) -> str:
    """Downloads GitHub repo as zip and extracts it"""
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, f"{p.urlparse(url).path.split('/')[-1]}.zip")

    with r.get(f"{url}/archive/refs/heads/main.zip", timeout=120) as req:
        with open(zip_path, "wb") as file:
            file.write(req.raise_for_status() or req.content)

    return temp_dir


def execute(url: str) -> str:
    """Main functions"""
    return url
