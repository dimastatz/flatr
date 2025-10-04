""" Tests for flattr module """
import os
import flatr.flatr


def test_execute():
    """Tests execute function"""
    assert flatr.flatr.execute("test") == "test"

    zip_file = flatr.flatr.download("https://github.com/dimastatz/flatr.git")
    assert zip_file.endswith(".zip")

    # Cleanup
    flatr.flatr.cleanup(zip_file)
    assert os.path.exists(zip_file) is False
