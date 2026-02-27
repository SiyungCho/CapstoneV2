Pickle file contains 2 different datasets (pandas dataframes), each with a timestamp, replay_mulitmodal can preview the data with the following:
```bash
python .\replay_multimodal_data.py --file=multimodal_data_1772131151.pkl --speed=10.0
```
the two datatypes are:
[eit_data] - from the wristband
[hand_data] - from the camera

# EIT DATA
contains:
timestamp
event - synch markers. look for measurements after Calibration for good data
phase - similar to event, ignore data taken pre-calibration, look for Measurement data
FrameID - form wristband, ID of EIT frame
INJ/Sense - which pair is driving current, which pair is reading it
Real/Img/Magnitude - the impedance values, in measurement these are delta Z, td-EIT measurements.
