# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "packaging>=26.0",
#     "requests>=2.33.1",
# ]
# ///

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests
from packaging.version import Version

_BASE_URL = "https://developer.apple.com/metal/cpp/files/"
_ARCHIVE_BASENAME = "metal-cpp"
_SCRIPT_PATH = Path(__file__).parent
_TARGET_PATH: Path = _SCRIPT_PATH / "ext" / "metal-cpp"
_VERSIONS: list[Version] = [Version("26"), Version("26.4")]
_VERSION_MAP = {v: _ARCHIVE_BASENAME + "_" + str(v) + ".zip" for v in _VERSIONS}


def main():
    already_downloaded = (_TARGET_PATH / "LICENSE.txt").exists()

    version = Version("26.4")
    try:
        filename = _VERSION_MAP[version]
    except KeyError:
        raise ValueError(f"no metal-cpp version {version}") from None

    url = _BASE_URL + filename
    content = requests.get(url)

    f = ZipFile(BytesIO(content.content))
    f.extractall(_TARGET_PATH)


if __name__ == "__main__":
    main()
