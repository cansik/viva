import os
from pathlib import Path
from typing import List, Union, Any

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


def path_serializer(obj: Any) -> str:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
