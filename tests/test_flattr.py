""" Tests for flattr module """
import flatr.flatr


def test_execute():
    """Tests execute function"""
    assert flatr.flatr.execute("test") == "test"
