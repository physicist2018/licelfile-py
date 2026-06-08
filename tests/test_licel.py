"""Tests for licelformat. Run with: pytest"""

import os

import numpy as np
import pytest

from licelformat import (
    LicelFile,
    LicelPack,
    LicelProfile,
    LoadLicelFile,
    NewLicelPack,
)

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "testdata")
TEST_FILE = os.path.join(TESTDATA_DIR, "b2651321.051986")
TMP_FILE = os.path.join(os.path.dirname(__file__), "..", "testdata", "_tmp_test.DAT")


class TestLicelProfile:
    """LicelProfile — header parsing, metadata roundtrip, scale_factor."""

    def test_parse_analog_channel(self):
        line = " 1 0 1 16380 1 0000 7.50 00355.o 0 0 00 000 12 002001 0.5000 BT0"
        p = LicelProfile(line)
        assert p.Active is True
        assert p.Photon is False
        assert p.LaserType == 1
        assert p.NDataPoints == 16380
        assert p.HighVoltage == 0
        assert p.BinWidth == 7.50
        assert p.Wavelength == 355.0
        assert p.Polarization == "o"
        assert p.BinShift == 0
        assert p.DecBinShift == 0
        assert p.AdcBits == 12
        assert p.NShots == 2001
        assert p.DiscrLevel == 0.5
        assert p.DeviceID == "BT"
        assert p.NCrate == 0

    def test_parse_photon_channel(self):
        line = " 1 1 1 16380 1 0000 7.50 00355.o 0 0 00 000 00 002001 3.1746 BC0"
        p = LicelProfile(line)
        assert p.Active is True
        assert p.Photon is True
        assert p.Wavelength == 355.0
        assert p.Polarization == "o"
        assert p.NShots == 2001
        assert p.DiscrLevel == 3.1746
        assert p.DeviceID == "BC"

    def test_metadata_roundtrip(self):
        """metadata() output can be parsed back."""
        line = " 1 0 1 16380 1 0000 7.50 00355.o 0 0 00 000 12 002001 0.5000 BT0"
        p = LicelProfile(line)
        meta = p.metadata()
        p2 = LicelProfile(meta)
        assert p2.Active == p.Active
        assert p2.Photon == p.Photon
        assert p2.Wavelength == p.Wavelength
        assert p2.NDataPoints == p.NDataPoints
        assert p2.NShots == p.NShots

    @pytest.mark.parametrize(
        "photon, adc_bits, nshots, discr, expected",
        [
            (False, 12, 2001, 0.5, 0.5 * 1000 / (4096 * 2001)),
            (True, 0, 2001, 0.0, 1.0 / (2001 * 0.05)),
        ],
    )
    def test_scale_factor(self, photon, adc_bits, nshots, discr, expected):
        p = LicelProfile()
        p.Photon = photon
        p.AdcBits = adc_bits
        p.NShots = nshots
        p.DiscrLevel = discr
        assert np.isclose(p.scale_factor(), expected)

    def test_to_dict(self):
        line = " 1 1 1 100 0 0000 7.50 00532.s 0 0 00 000 00 002001 3.1746 BC0"
        p = LicelProfile(line)
        d = p.to_dict()
        assert d["wavelength"] == 532.0
        assert d["is_photon"] is True
        assert d["n_shots"] == 2001
        assert "data" in d


class TestLicelFile:
    """LicelFile — loading, selection, round‑trip save/reload."""

    def test_load(self):
        a = LoadLicelFile(TEST_FILE)
        assert a.FileLoaded is True
        assert a.MeasurementSite == "Vladivos"
        assert a.AltitudeAboveSeaLevel == 20.0
        assert a.Longitude == 131.9
        assert a.Latitude == 43.1
        assert a.Zenith == 50.0
        assert a.Laser1NShots == 2001
        assert a.Laser1Freq == 20
        assert a.Laser2NShots == 0
        assert a.Laser2Freq == 10
        assert a.NDatasets == 12
        assert len(a.Profiles) == 12

    def test_datetime(self):
        a = LoadLicelFile(TEST_FILE)
        assert a.MeasurementStartTime.year == 2026
        assert a.MeasurementStartTime.month == 5
        assert a.MeasurementStartTime.day == 13
        assert a.MeasurementStartTime.hour == 21
        assert a.MeasurementStartTime.minute == 3
        assert a.MeasurementStartTime.second == 45

    def test_select_certain_wavelength_found(self):
        a = LoadLicelFile(TEST_FILE)
        p = a.select_certain_wavelength(True, 355.0)
        assert p.Wavelength == 355.0
        assert p.Photon is True

    def test_select_certain_wavelength_not_found(self):
        a = LoadLicelFile(TEST_FILE)
        p = a.select_certain_wavelength(True, 999.0)
        assert p.Wavelength == 0.0
        assert p.Active is False

    def test_all_profiles_present(self):
        a = LoadLicelFile(TEST_FILE)
        wavelengths = [(p.Wavelength, p.Photon) for p in a.Profiles]
        expected = [
            (355.0, False),
            (355.0, True),
            (353.0, False),
            (353.0, True),
            (530.0, False),
            (530.0, True),
            (532.0, False),
            (532.0, True),
            (532.0, False),
            (532.0, True),
            (1064.0, False),
            (408.0, True),
        ]
        assert wavelengths == expected

    def test_save_roundtrip(self):
        """Save → reload must produce identical scaled data."""
        a = LoadLicelFile(TEST_FILE)
        try:
            a.save(TMP_FILE + "1")
            sut_path = TMP_FILE + "1"
            b = LoadLicelFile(sut_path)
            assert b.FileLoaded is True
            assert b.NDatasets == a.NDatasets
            for i in range(len(a.Profiles)):
                assert np.allclose(
                    a.Profiles[i].Data, b.Profiles[i].Data, rtol=1e-12
                ), f"Profile {i} differs after round‑trip"
        finally:
            for suffix in ("", "1"):
                p = TMP_FILE + suffix
                if os.path.exists(p):
                    os.remove(p)

    def test_to_dict(self):
        a = LoadLicelFile(TEST_FILE)
        d = a.to_dict()
        assert d["location"] == "Vladivos"
        assert d["start_time"] is not None
        assert d["dataset_count"] == 12
        assert len(d["datasets"]) == 12
        assert d["datasets"][0]["wavelength"] == 355.0

    def test_filter_all_match(self):
        """filter with predicate that always returns True returns all profiles."""
        a = LoadLicelFile(TEST_FILE)
        result = a.filter(lambda p: True)
        assert len(result) == len(a.Profiles)
        assert result == a.Profiles

    def test_filter_none_match(self):
        """filter with predicate that always returns False returns empty list."""
        a = LoadLicelFile(TEST_FILE)
        result = a.filter(lambda p: False)
        assert result == []

    def test_filter_photon_channels(self):
        """filter selects only photon‑counting profiles."""
        a = LoadLicelFile(TEST_FILE)
        result = a.filter(lambda p: p.Photon)
        assert len(result) > 0
        assert all(p.Photon for p in result)

    def test_filter_analog_channels(self):
        """filter selects only analog profiles."""
        a = LoadLicelFile(TEST_FILE)
        result = a.filter(lambda p: not p.Photon)
        assert len(result) > 0
        assert all(not p.Photon for p in result)

    def test_filter_by_wavelength(self):
        """filter selects only profiles at a given wavelength."""
        a = LoadLicelFile(TEST_FILE)
        result = a.filter(lambda p: p.Wavelength == 532.0)
        assert len(result) > 0
        assert all(p.Wavelength == 532.0 for p in result)

    def test_filter_combined_predicate(self):
        """filter with a combined predicate (wavelength + type)."""
        a = LoadLicelFile(TEST_FILE)
        # Photon channels at 355 nm
        result = a.filter(lambda p: p.Wavelength == 355.0 and p.Photon)
        assert len(result) == 1
        assert result[0].Wavelength == 355.0
        assert result[0].Photon is True

    def test_filter_active_only(self):
        """filter selects only active profiles."""
        a = LoadLicelFile(TEST_FILE)
        result = a.filter(lambda p: p.Active)
        assert all(p.Active for p in result)

    def test_filter_returns_list_of_profiles(self):
        """filter returns a list of LicelProfile instances."""
        a = LoadLicelFile(TEST_FILE)
        result = a.filter(lambda p: p.Wavelength == 355.0)
        assert isinstance(result, list)
        assert all(isinstance(p, LicelProfile) for p in result)


class TestLicelPack:
    """LicelPack — multi‑file loading and wavelength selection."""

    def test_new_licel_pack_single_file(self):
        pack = NewLicelPack(TEST_FILE)
        assert len(pack.Data) == 1
        assert pack.StartTime is not None
        assert pack.StartTime.year == 2026

    def test_new_licel_pack_empty_mask(self):
        with pytest.raises(FileNotFoundError):
            NewLicelPack("nonexistent_*.DAT")

    def test_select_wavelength_from_pack(self):
        pack = NewLicelPack(TEST_FILE)
        profiles = pack.select_certain_wavelength(True, 355.0)
        assert len(profiles) == 1
        assert profiles[0].Wavelength == 355.0
        assert profiles[0].Photon is True

    def test_select_wavelength_not_found(self):
        pack = NewLicelPack(TEST_FILE)
        profiles = pack.select_certain_wavelength(True, 999.0)
        assert profiles == []

    def test_to_dict(self):
        pack = NewLicelPack(TEST_FILE)
        d = pack.to_dict()
        assert d["start_time"] is not None
        assert len(d["data"]) == 1

    def test_filter_profiles(self):
        """filter collects all profiles from matching files."""
        pack = NewLicelPack(TEST_FILE)
        result = pack.filter(lambda p: True)
        assert isinstance(result, list)
        expected = sum(len(lf.Profiles) for lf in pack.Data.values())
        assert len(result) == expected

    def test_filter_profiles_none(self):
        """filter with False predicate returns empty list."""
        pack = NewLicelPack(TEST_FILE)
        result = pack.filter(lambda p: False)
        assert result == []

    def test_filter_profiles_photon(self):
        """filter collects only photon profiles."""
        pack = NewLicelPack(TEST_FILE)
        result = pack.filter(lambda p: p.Photon)
        for profile in result:
            assert profile.Photon is True

    def test_filter_files_all_match(self):
        """filter_files with predicate that always returns True returns full copy."""
        pack = NewLicelPack(TEST_FILE)
        result = pack.filter_files(lambda lf: True)
        assert isinstance(result, LicelPack)
        assert len(result.Data) == len(pack.Data)
        assert result.StartTime == pack.StartTime
        assert result.StopTime == pack.StopTime

    def test_filter_files_none_match(self):
        """filter_files with predicate that always returns False returns empty pack."""
        pack = NewLicelPack(TEST_FILE)
        result = pack.filter_files(lambda lf: False)
        assert isinstance(result, LicelPack)
        assert len(result.Data) == 0
        assert result.StartTime is None
        assert result.StopTime is None

    def test_filter_files_by_site(self):
        """filter_files selects files by measurement site."""
        pack = NewLicelPack(TEST_FILE)
        site = pack.Data[list(pack.Data.keys())[0]].MeasurementSite
        result = pack.filter_files(lambda lf: lf.MeasurementSite == site)
        assert len(result.Data) == 1
        result = pack.filter_files(lambda lf: lf.MeasurementSite == "Nonexistent")
        assert len(result.Data) == 0

    def test_filter_files_returns_new_pack(self):
        """filter_files returns a new LicelPack, not the same object."""
        pack = NewLicelPack(TEST_FILE)
        result = pack.filter_files(lambda lf: True)
        assert result is not pack

    def test_filter_files_preserves_references(self):
        """filter_files shares the same LicelFile objects (shallow copy)."""
        pack = NewLicelPack(TEST_FILE)
        result = pack.filter_files(lambda lf: True)
        for key in result.Data:
            assert result.Data[key] is pack.Data[key]


class TestSubtractBackground:
    """Subtract background — mean, median, dark."""

    @pytest.fixture
    def profile(self):
        p = LicelProfile()
        p.BinWidth = 10.0
        p.NDataPoints = 100
        p.Data = [float(i) for i in range(100)]
        return p

    @pytest.fixture
    def analog_profile(self):
        p = LicelProfile()
        p.BinWidth = 10.0
        p.NDataPoints = 100
        p.Data = [float(i) for i in range(100)]
        p.DeviceID = "BT"
        p.Photon = False
        p.Wavelength = 355.0
        p.Polarization = "o"
        return p

    @pytest.fixture
    def photon_profile(self):
        p = LicelProfile()
        p.BinWidth = 10.0
        p.NDataPoints = 100
        p.Data = [float(i) for i in range(100)]
        p.DeviceID = "BC"
        p.Photon = True
        p.Wavelength = 355.0
        p.Polarization = "o"
        return p

    # --- LicelProfile tests ---

    def test_profile_subtract_mean(self, profile):
        profile.subtract_background(method="mean", bgrRange=800.0)
        bg = 89.5
        assert abs(profile.Data[0] - (0 - bg)) < 1e-9
        assert abs(profile.Data[50] - (50 - bg)) < 1e-9
        assert abs(profile.Data[99] - (99 - bg)) < 1e-9

    def test_profile_subtract_median(self, profile):
        profile.subtract_background(method="median", bgrRange=800.0)
        bg = 89.5
        assert abs(profile.Data[0] - (0 - bg)) < 1e-9

    def test_profile_subtract_dark(self, profile):
        dark = LicelProfile()
        dark.Data = [1.0] * 100
        profile.subtract_background(method="dark", dark_profile=dark)
        assert profile.Data[0] == -1.0
        assert profile.Data[50] == 49.0
        assert profile.Data[99] == 98.0

    def test_profile_subtract_mean_bgrange_too_large(self, profile):
        with pytest.raises(ValueError, match="beyond profile data"):
            profile.subtract_background(method="mean", bgrRange=2000.0)

    def test_profile_subtract_dark_missing(self, profile):
        with pytest.raises(ValueError, match="dark_profile is required"):
            profile.subtract_background(method="dark")

    def test_profile_subtract_dark_length_mismatch(self, profile):
        dark = LicelProfile()
        dark.Data = [1.0] * 50
        with pytest.raises(ValueError, match="data length does not match"):
            profile.subtract_background(method="dark", dark_profile=dark)

    def test_profile_subtract_unknown_method(self, profile):
        with pytest.raises(ValueError, match="Unknown method: invalid"):
            profile.subtract_background(method="invalid", bgrRange=800.0)

    # --- LicelFile tests ---

    def test_file_subtract_mean(self, analog_profile, photon_profile):
        lf = LicelFile()
        lf.Profiles = [analog_profile, photon_profile]
        lf.NDatasets = 2
        lf.subtract_background(method="mean", bgrRange=800.0)
        bg = 89.5
        assert abs(lf.Profiles[0].Data[0] - (0 - bg)) < 1e-9
        assert abs(lf.Profiles[1].Data[0] - (0 - bg)) < 1e-9

    def test_file_subtract_dark(self, analog_profile, photon_profile):
        lf = LicelFile()
        lf.Profiles = [analog_profile, photon_profile]
        lf.NDatasets = 2

        dark_analog = LicelProfile()
        dark_analog.Data = [10.0] * 100
        dark_analog.DeviceID = "BT"
        dark_analog.Photon = False
        dark_analog.Wavelength = 355.0
        dark_analog.Polarization = "o"

        dark_photon = LicelProfile()
        dark_photon.Data = [5.0] * 100
        dark_photon.DeviceID = "BC"
        dark_photon.Photon = True
        dark_photon.Wavelength = 355.0
        dark_photon.Polarization = "o"

        dark_file = LicelFile()
        dark_file.Profiles = [dark_analog, dark_photon]

        lf.subtract_background(method="dark", dark_file=dark_file)
        assert lf.Profiles[0].Data[0] == -10.0
        assert lf.Profiles[1].Data[0] == -5.0

    # --- LicelPack tests ---

    def test_pack_subtract_mean(self):
        lf = LicelFile()
        p = LicelProfile()
        p.BinWidth = 10.0
        p.NDataPoints = 100
        p.Data = [float(i) for i in range(100)]
        lf.Profiles = [p]
        lf.NDatasets = 1

        pack = LicelPack()
        pack.Data["test"] = lf

        pack.subtract_background(method="mean", bgrRange=800.0)
        bg = 89.5
        assert abs(pack.Data["test"].Profiles[0].Data[0] - (0 - bg)) < 1e-9


class TestAverage:
    """LicelPack.average() — averaging profiles across files."""

    def _make_file(self, site: str, data_sets: list, start_time=None):
        """Create a LicelFile with profiles from data_sets.

        Args:
            site: Measurement site name.
            data_sets: list of lists, each inner list is raw data for one profile.

        Returns:
            A LicelFile with the specified profiles.
        """
        lf = LicelFile()
        lf.MeasurementSite = site
        for raw in data_sets:
            p = LicelProfile()
            p.BinWidth = 10.0
            p.NDataPoints = len(raw)
            p.Data = [float(v) for v in raw]
            p.Wavelength = 355.0
            p.Polarization = "o"
            p.DeviceID = "BC"
            p.Photon = True
            lf.Profiles.append(p)
        lf.NDatasets = len(data_sets)
        return lf

    def test_average_two_files(self):
        f1 = self._make_file("site1", [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        f2 = self._make_file("site2", [[3.0, 4.0, 5.0], [30.0, 40.0, 50.0]])

        pack = LicelPack()
        pack.Data["f1"] = f1
        pack.Data["f2"] = f2

        avg = pack.average()
        assert avg.NDatasets == 2
        assert len(avg.Profiles) == 2

        # Profile 0: (1+3)/2=2, (2+4)/2=3, (3+5)/2=4
        np.testing.assert_array_almost_equal(avg.Profiles[0].Data, [2.0, 3.0, 4.0])
        # Profile 1: (10+30)/2=20, (20+40)/2=30, (30+50)/2=40
        np.testing.assert_array_almost_equal(avg.Profiles[1].Data, [20.0, 30.0, 40.0])

        # Metadata from first file
        assert avg.MeasurementSite == "site1"
        assert avg.Profiles[0].DeviceID == "BC"

    def test_average_three_files(self):
        f1 = self._make_file("s", [[1.0, 2.0, 3.0]])
        f2 = self._make_file("s", [[4.0, 5.0, 6.0]])
        f3 = self._make_file("s", [[7.0, 8.0, 9.0]])

        pack = LicelPack()
        pack.Data["f1"] = f1
        pack.Data["f2"] = f2
        pack.Data["f3"] = f3

        avg = pack.average()
        # (1+4+7)/3=4, (2+5+8)/3=5, (3+6+9)/3=6
        np.testing.assert_array_almost_equal(avg.Profiles[0].Data, [4.0, 5.0, 6.0])

    def test_average_empty_pack(self):
        pack = LicelPack()
        with pytest.raises(ValueError, match="empty"):
            pack.average()

    def test_average_mismatch_nprofiles(self):
        f1 = self._make_file("s", [[1.0, 2.0]])
        f2 = self._make_file("s", [[3.0, 4.0], [5.0, 6.0]])
        f2.NDatasets = 2

        pack = LicelPack()
        pack.Data["f1"] = f1
        pack.Data["f2"] = f2

        with pytest.raises(ValueError, match="profiles"):
            pack.average()

    def test_average_truncates_to_shortest(self):
        f1 = self._make_file("s", [[1.0, 2.0, 3.0, 4.0]])
        f2 = self._make_file("s", [[5.0, 6.0, 7.0]])
        f2.Profiles[0].NDataPoints = 3
        f2.Profiles[0].Data = [5.0, 6.0, 7.0]

        pack = LicelPack()
        pack.Data["f1"] = f1
        pack.Data["f2"] = f2

        avg = pack.average()
        # min NDataPoints = 3, so (1+5)/2=3, (2+6)/2=4, (3+7)/2=5
        assert avg.Profiles[0].NDataPoints == 3
        np.testing.assert_array_almost_equal(avg.Profiles[0].Data, [3.0, 4.0, 5.0])
