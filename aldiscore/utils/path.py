from pathlib import Path, PurePath, PosixPath
import os


import os
from pathlib import PosixPath, WindowsPath

BasePath = PosixPath if os.name == "posix" else WindowsPath


class WildcardPath(BasePath):
    """Extends pathlib.Path to allow for easy wildcard formatting."""

    def format(self, **wildcards: dict[str, str]):
        formatted_paths = []
        for seg in self.parts:
            formatted_paths.append(seg.format(**wildcards))
        return WildcardPath(*formatted_paths)

    def listdir(self, dirs_only=False, suffix=None):
        contents = os.listdir(self)
        if dirs_only:
            contents = list(filter(lambda c: os.path.isdir(self / c), contents))
        if suffix:
            contents = list(filter(lambda c: c.endswith(suffix), contents))
        return contents
