"""
LicelPack — collection of Licel measurements loaded by file mask or from ZIP.

Provides loading and querying functionality for multiple Licel files.
"""

import glob
import io
import re
import zipfile
from datetime import datetime
from typing import Dict, List, Optional

from .licelfile import (
    LicelFile,
    LicelProfilesList,
    LoadLicelFile,
    LoadLicelFileFromReader,
)


def _is_valid_filename(filename: str) -> bool:
    """Check if filename matches the pattern 'b*.*'."""
    return bool(re.match(r"^[a-z].*\..+", filename))


class LicelPack:
    """Collection of Licel measurements."""

    __slots__ = ("StartTime", "Data")

    def __init__(self):
        self.StartTime: Optional[datetime] = None
        self.Data: Dict[str, LicelFile] = {}

    def select_certain_wavelength(
        self, is_photon: bool, wavelength: float
    ) -> LicelProfilesList:
        """Select profiles by wavelength and type from all files in the pack."""
        result: LicelProfilesList = []
        for licf in self.Data.values():
            profile = licf.select_certain_wavelength(is_photon, wavelength)
            if profile.Wavelength != 0:
                result.append(profile)
        return result

    def save(self) -> None:
        """Save all files in the pack to disk."""
        for fname, licf in self.Data.items():
            licf.save(fname)

    def save_to_zip(
        self,
        zip_path: str,
        compression: int = zipfile.ZIP_DEFLATED,
        compresslevel: int = 6,
    ) -> None:
        """Save all files in the pack to a ZIP archive.

        Args:
            zip_path: Path to the output ZIP file.
            compression: Compression method (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, etc.).
            compresslevel: Compression level (0-9). Default 6.
                          - 0: no compression (store only)
                          - 1: fastest, least compression
                          - 6: default balance
                          - 9: slowest, most compression

        Raises:
            ValueError: If compresslevel is outside the valid range for the chosen method.
        """
        with zipfile.ZipFile(
            zip_path, "w", compression=compression, compresslevel=compresslevel
        ) as zw:
            for fname, licf in self.Data.items():
                # Strip leading slash if present (used in NewLicelPackFromZip)
                arcname = fname.lstrip("/")
                zw.writestr(arcname, licf.to_bytes(arcname))

    def __repr__(self) -> str:
        return f"LicelPack(start={self.StartTime}, files={len(self.Data)})"

    def to_dict(self) -> dict:
        """Convert LicelPack to a dictionary (for JSON serialization)."""
        return {
            "start_time": (self.StartTime.isoformat() if self.StartTime else None),
            "data": {name: lf.to_dict() for name, lf in self.Data.items()},
        }


def NewLicelPack(mask: str) -> LicelPack:
    """Load files according to a glob mask."""
    pack = LicelPack()
    files = glob.glob(mask)
    if not files:
        raise FileNotFoundError(f"No files found matching mask: {mask}")

    for i, fname in enumerate(files):
        pack.Data[fname] = LoadLicelFile(fname)
        if i == 0:
            pack.StartTime = pack.Data[fname].MeasurementStartTime

    return pack


def NewLicelPackFromZip(zip_path: str) -> LicelPack:
    """Load files from a ZIP archive."""
    pack = LicelPack()

    with zipfile.ZipFile(zip_path, "r") as zr:
        for info in zr.infolist():
            fname = info.filename
            if not _is_valid_filename(fname):
                continue

            # Read file contents into memory
            file_bytes = zr.read(fname)

            # Load LicelFile from bytes
            licf = LoadLicelFileFromReader(io.BytesIO(file_bytes), len(file_bytes))

            full_path = "/" + fname
            pack.Data[full_path] = licf

            if len(pack.Data) == 1:
                pack.StartTime = licf.MeasurementStartTime

    return pack
