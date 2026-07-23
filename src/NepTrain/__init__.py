"""NepTrain package metadata."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("NepTrain")
except PackageNotFoundError:
    __version__ = "0+unknown"


__all__ = ["__version__"]
