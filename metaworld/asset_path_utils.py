"""Set of utilities for retrieving asset paths for the environments."""

from pathlib import Path

_CURRENT_FILE_DIR = Path(__file__).parent.absolute()

ENV_ASSET_DIR_V3 = _CURRENT_FILE_DIR / "assets"


def full_v3_path_for(file_name: str) -> str:
    """Convert a relative asset file name to its full path.

    Args:
        file_name: Name of the asset file. Can include subdirectories.

    Returns:
        The full path to the asset file.

    """
    return str(ENV_ASSET_DIR_V3 / file_name)
