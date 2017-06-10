# cs224s-turn-taking
Final project for CS 224S (Spoken Language Processing)

### Data

switchboard_sample: https://archive.org/details/SwitchboardCorpusSample

swb_ms98_transcriptions: https://www.isip.piconepress.com/projects/switchboard/

Switchboard - /afs/ir/data/linguistic-data/Switchboard/Audio-swbd1ph2

Penn Treebank 3 Annotations - /afs/ir/data/linguistic-data/Treebank/LDC99T42_Treebank-3/dysfl/dff/swbd

SWDA - http://compprag.christopherpotts.net/swda.html#download 

Time-stamped transcriptions - /afs/ir/data/linguistic-data/Switchboard/SWBD-MSState-Transcripts/swb_ms98_transcriptions

### Files

baseline.py: Baseline experiment

turn_labels.py: Aligns the time-stamped word transcript and disfluency transcript, labels filler words and discourse markers

extract_clips.py: Extracts clips from Switchboard audio

extract_features.py: Extracts features from clips

rnn_classifier.py: RNN sequence classifier for features, created using RNN Tensorflow tutorials written by Danijar Hafner (https://danijar.com)

swda.py: created by Christopher Potts, Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License
