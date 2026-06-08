"""
LicelFormat — Python port of the Go licelformat package.

Provides utilities for parsing and processing Licel format data files.
Supports reading, extracting metadata, and converting binary data into
usable formats. Works with Licel files containing measurement profiles
and other associated data.
"""

from .licelfile import (
    LicelFile,
    LicelProfilesList,
    LoadLicelFile,
    LoadLicelFileFromReader,
)
from .licelpack import LicelPack, NewLicelPack, NewLicelPackFromZip
from .licelprofile import LICEL_MAX_RESERVED, LicelProfile

__all__ = [
    "LicelProfile",
    "LICEL_MAX_RESERVED",
    "LicelFile",
    "LicelProfilesList",
    "LoadLicelFile",
    "LoadLicelFileFromReader",
    "LicelPack",
    "NewLicelPack",
    "NewLicelPackFromZip",
]
