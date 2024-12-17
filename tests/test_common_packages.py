from pipelines.common import PACKAGES, packages


def test_packages_returns_package_version():
    result = packages("keras")
    assert result["keras"] == PACKAGES["keras"]


def test_packages_returns_package_without_version():
    result = packages("random-package")
    assert result["random-package"] == ""


def test_packages_returns_multiple_packages():
    result = packages("keras", "numpy")
    assert "keras" in result
    assert "numpy" in result
