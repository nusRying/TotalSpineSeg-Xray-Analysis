from importlib.metadata import PackageNotFoundError, metadata


try:
    package_metadata = metadata("totalspineseg")
except PackageNotFoundError:
    package_metadata = None


if package_metadata is None:
    ZIP_URLS = {}
    VERSION = "0+local"
else:
    ZIP_URLS = dict(
        [meta.split(", ") for meta in package_metadata.get_all("Project-URL") if meta.startswith("Dataset")]
    )
    VERSION = package_metadata.get("version")
