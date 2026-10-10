"""CZI metadata parsing (readers/czi.py)."""

import os
import xml.etree.ElementTree as ET

import pytest

from pyvistra.readers.czi import CZIReader

CZI_SAMPLE = os.path.join(os.path.dirname(__file__), "data", "LSM980_obj63x_B1_d20220324.czi")

# Same nesting as a real LSM980 file: the acquisition setup's track
# channels (including inactive detectors) precede the image's own list.
_XML = """
<ImageDocument><Metadata>
  <Experiment><ExperimentBlocks><AcquisitionBlock><SubDimensionSetups>
    <MultiTrackSetup><Track><Channels>
      <Channel Name="Ch1"/><Channel Name="DAPI"/><Channel Name="T-PMT"/>
    </Channels></Track></MultiTrackSetup>
  </SubDimensionSetups></AcquisitionBlock></ExperimentBlocks></Experiment>
  <Information><Image><Dimensions><Channels>
    <Channel Name="DAPI"><EmissionWavelength>465</EmissionWavelength>
      <ExcitationWavelength>353</ExcitationWavelength></Channel>
    <Channel Name="GFP"><EmissionWavelength>509</EmissionWavelength></Channel>
  </Channels></Dimensions></Image></Information>
  <DisplaySetting><Channels><Channel Name="DAPI"/><Channel Name="GFP"/></Channels></DisplaySetting>
</Metadata></ImageDocument>
"""


def test_channels_come_from_image_dimensions_not_acquisition_tracks():
    """Regression: `.//Channel` picked up acquisition-track detectors first,
    so channel 0 of a DAPI image was named "Ch1" with no wavelengths."""
    reader = CZIReader.__new__(CZIReader)  # skip file I/O; parse XML only
    reader.n_channels = 2

    reader._parse_channels(ET.fromstring(_XML))

    assert [(c["name"], c["emission_wavelength"]) for c in reader.channels_info] == [
        ("DAPI", 465.0),
        ("GFP", 509.0),
    ]
    assert reader.channels_info[0]["excitation_wavelength"] == 353.0


def test_real_lsm980_file_channel_metadata():
    pytest.importorskip("aicspylibczi")
    if not os.path.exists(CZI_SAMPLE):
        pytest.skip("sample CZI not present")

    with CZIReader(CZI_SAMPLE) as reader:
        assert reader.channels_info == [
            {"id": 0, "name": "DAPI", "emission_wavelength": 465.0, "excitation_wavelength": 353.0}
        ]
