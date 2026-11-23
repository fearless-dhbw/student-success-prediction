"""Download the official UCI dataset into the ignored raw-data directory."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from student_success.data import PROJECT_ROOT

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "uci_697_source.zip"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urlretrieve(SOURCE_URL, ZIP_PATH)
    with ZipFile(ZIP_PATH) as archive:
        archive.extractall(RAW_DIR / "uci_697_source")
    print(f"Downloaded official UCI dataset to {RAW_DIR / 'uci_697_source' / 'data.csv'}")


if __name__ == "__main__":
    main()
