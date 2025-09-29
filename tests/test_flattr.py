""" Tests for flattr module """
import flattr.flattr


def test_execute():
    """Tests execute function"""
    assert flattr.flattr.execute("test") == "test"
