"""
LicelProfile — measurement channel structure.

Represents a single measurement channel (profile) in a Licel file.
"""

import struct
from typing import List

LICEL_MAX_RESERVED = 3


def _str2bool(s: str) -> bool:
    """Convert string to boolean."""
    return s.lower() in ("1", "true", "yes")


def _str2int(s: str) -> int:
    """Convert string to integer."""
    return int(s)


def _str2float(s: str) -> float:
    """Convert string to float."""
    return float(s)


def _btoi(b: bool) -> int:
    """Convert boolean to integer (0/1)."""
    return 1 if b else 0


class LicelProfile:
    """Represents a single measurement channel in a Licel file."""

    __slots__ = (
        "Active",
        "Photon",
        "LaserType",
        "NDataPoints",
        "Reserved",
        "HighVoltage",
        "BinWidth",
        "Wavelength",
        "Polarization",
        "BinShift",
        "DecBinShift",
        "AdcBits",
        "NShots",
        "DiscrLevel",
        "DeviceID",
        "NCrate",
        "Data",
    )

    def __init__(self, line: str = None):
        self.Active: bool = False
        self.Photon: bool = False
        self.LaserType: int = 0
        self.NDataPoints: int = 0
        self.Reserved: List[int] = [0, 0, 0]
        self.HighVoltage: int = 0
        self.BinWidth: float = 0.0
        self.Wavelength: float = 0.0
        self.Polarization: str = ""
        self.BinShift: int = 0
        self.DecBinShift: int = 0
        self.AdcBits: int = 0
        self.NShots: int = 0
        self.DiscrLevel: float = 0.0
        self.DeviceID: str = ""
        self.NCrate: int = 0
        self.Data: List[float] = []

        if line is not None:
            self._parse(line)

    def _parse(self, line: str) -> None:
        """Parse a string line into LicelProfile."""
        items = line.split()
        wvlpol = items[7].split(".", 1)

        self.Active = _str2bool(items[0])
        self.Photon = _str2bool(items[1])
        self.LaserType = _str2int(items[2])
        self.NDataPoints = _str2int(items[3])
        self.Reserved = [
            _str2int(items[4]),
            _str2int(items[8]),
            _str2int(items[9]),
        ]
        self.HighVoltage = _str2int(items[5])
        self.BinWidth = _str2float(items[6])
        self.Wavelength = _str2float(wvlpol[0])
        self.Polarization = wvlpol[1] if len(wvlpol) > 1 else ""
        self.BinShift = _str2int(items[10])
        self.DecBinShift = _str2int(items[11])
        self.AdcBits = _str2int(items[12])
        self.NShots = _str2int(items[13])
        self.DiscrLevel = _str2float(items[14])
        self.DeviceID = items[15][:2]
        self.NCrate = _str2int(items[15][2:])

    def metadata(self) -> str:
        """Return the metadata string for this profile."""
        if self.Photon:
            s = (
                f" {_btoi(self.Active):1d} {_btoi(self.Photon):1d} {self.LaserType:1d} "
                f"{self.NDataPoints:05d} {self.Reserved[0]:1d} {self.HighVoltage:04d} "
                f"{self.BinWidth:04.2f} {int(self.Wavelength):05d}.{self.Polarization:<1s} "
                f"{0:1d} {0:1d} {self.BinShift:02d} {self.DecBinShift:03d} "
                f"{self.AdcBits:02d} {self.NShots:06d} {self.DiscrLevel:05.4f} "
                f"{self.DeviceID:2s}{self.NCrate:01d}"
            )
        else:
            s = (
                f" {_btoi(self.Active):1d} {_btoi(self.Photon):1d} {self.LaserType:1d} "
                f"{self.NDataPoints:05d} {self.Reserved[0]:1d} {self.HighVoltage:04d} "
                f"{self.BinWidth:04.2f} {int(self.Wavelength):05d}.{self.Polarization:<1s} "
                f"{0:1d} {0:1d} {self.BinShift:02d} {self.DecBinShift:03d} "
                f"{self.AdcBits:02d} {self.NShots:06d} {self.DiscrLevel:05.3f} "
                f"{self.DeviceID:2s}{self.NCrate:01d}"
            )
        return f"{s:<78s}\r\n"

    def scale_factor(self) -> float:
        """Return the scaling factor that was applied during loading.

        Analog channels: scale = DiscrLevel * 1000 / (2^AdcBits * NShots)
        Photon channels: scale = 1 / (NShots * 0.05)
        """
        if not self.Photon:
            adc_scale = 1 << self.AdcBits
            return self.DiscrLevel * 1000.0 / float(adc_scale * self.NShots)
        else:
            return 1.0 / (float(self.NShots) * 0.05)

    def profile(self) -> bytes:
        """Convert profile data to binary bytes (little-endian int32).

        Applies unscale first: raw = round(scaled_value / scale_factor),
        then packs as little-endian int32, matching the original file format.
        """
        inv_scale = 1.0 / self.scale_factor() if self.scale_factor() != 0.0 else 1.0
        buf = b""
        for num in self.Data:
            buf += struct.pack("<i", round(num * inv_scale))
        return buf + b"\r\n"

    def __repr__(self) -> str:
        return (
            f"LicelProfile(Active={self.Active}, Photon={self.Photon}, "
            f"Wavelength={self.Wavelength}, NDataPoints={self.NDataPoints}, "
            f"NShots={self.NShots}, Polarization={self.Polarization})"
        )

    def to_dict(self) -> dict:
        """Convert profile to a dictionary (for JSON serialization)."""
        return {
            "is_active": self.Active,
            "is_photon": self.Photon,
            "laser_type": self.LaserType,
            "data_points": self.NDataPoints,
            "reserved": self.Reserved,
            "high_voltage": self.HighVoltage,
            "bin_width": self.BinWidth,
            "wavelength": self.Wavelength,
            "polarization": self.Polarization,
            "bin_shift": self.BinShift,
            "dec_bin_shift": self.DecBinShift,
            "adc_bits": self.AdcBits,
            "n_shots": self.NShots,
            "discr_level": self.DiscrLevel,
            "device_id": self.DeviceID,
            "n_crate": self.NCrate,
            "data": self.Data,
        }
