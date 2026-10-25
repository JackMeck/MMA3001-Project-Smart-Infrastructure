import smart_infrastructure


def test_package_import():
    assert callable(getattr(smart_infrastructure, "project", None)) or hasattr(smart_infrastructure, "project")
