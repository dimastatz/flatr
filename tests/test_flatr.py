""" Tests for flattr module """
import os
import flatr.flatr


def test_execute():
    """Tests execute function"""
    assert flatr.flatr.execute("test") == "test"

    zip_file = flatr.flatr.download("https://github.com/dimastatz/flatr.git")
    assert zip_file.endswith(".zip")

    extract_path = flatr.flatr.unzip(zip_file)
    assert os.path.exists(extract_path)

    # Cleanup
    flatr.flatr.cleanup(zip_file)
    assert os.path.exists(zip_file) is False
    assert os.path.exists(extract_path) is False
