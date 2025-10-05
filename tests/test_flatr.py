""" Tests for flattr module """
import os
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
    assert any("README" in f for f in files)
    assert any(f.endswith(".py") for f in files)

    markdown_path = os.path.join(extract_path, "output.md")
    flatr.flatr.write_markdown(files, repo_url, markdown_path)
    assert os.path.exists(markdown_path)

    # Cleanup
    flatr.flatr.cleanup(zip_file)
    assert os.path.exists(zip_file) is False
    assert os.path.exists(extract_path) is False
