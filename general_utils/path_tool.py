from pathlib import Path
from typing import TypeVar, Any, Set, Callable, List, Iterable

PROJECT_PATH = str(Path(__file__).parents[1])
RUN_PATH = str(Path(PROJECT_PATH) / "runs")

DATA_PATH = str(Path(PROJECT_PATH) / ".data")

Local_DATAPATH = str(Path(__file__).parents[2] / "DataBase")
T_path = TypeVar("T_path", str, Path)


def path2Path(path: T_path) -> Path:
    assert isinstance(path, (Path, str)), type(path)
    return Path(path) if isinstance(path, str) else path
