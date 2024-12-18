import os
from pathlib import Path
from typing import List, Union

Pathable = Union[str, os.PathLike]


def get_files(path: Pathable, *filters: str, recursive: bool = False) -> List[Path]:
    if len(filters) == 0:
        filters = ["*"]  # Default to matching all files

    path = Path(path)  # Convert path to a Path object
    method = path.rglob if recursive else path.glob  # Choose between glob or rglob based on 'recursive'

    files = []
    for filter_pattern in filters:
        files.extend(method(pattern=filter_pattern))

    return list(sorted(set(files)))
