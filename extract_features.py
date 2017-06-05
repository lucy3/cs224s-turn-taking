from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import numpy as np
import collections
import os
from pydub import AudioSegment
AudioSegment.ffmpeg = "/usr/bin/ffmpeg"
 
CLIPS_DIR_PATH = './clips/clips/' 
FEATURE_PICKLES = './feat_pickles/'

def extractFeatures():
    files = []
    print "Getting example names..."
    with open('./labels') as f:
        for line in f:
            files.append("_".join(line.split("\t")[0].split("_")[:-1])+'.wav')
    f.close()
    
    print "Walking through files to extract features..." 
    for dirpath, dirnames, filenames in os.walk(CLIPS_DIR_PATH):
        for f in filenames:
            if f in files:
		print "Extracting for", f
                path = './clips/clips/' + f
                audiofile = AudioSegment.from_file(path)
                data = np.fromstring(audiofile._data, np.int16)
                Fs = audiofile.frame_rate
                x = []
                for chn in xrange(audiofile.channels):
                    x.append(data[chn::audiofile.channels])
                x = np.array(x).T
		if x.ndim==2:
        	    if x.shape[1]==1:
            	        x = x.flatten()
                try:
                    features = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs).T
                    np.save(FEATURE_PICKLES + f, features)
		except ValueError as e:
		    print e

extractFeatures()
