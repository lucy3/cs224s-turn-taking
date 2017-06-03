from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import numpy as np
import collections

CLIPS_DIR_PATH = './clips/' 
files = []

def extractFeatures():
    with open('./labels') as f:
        for line in f:
            files.append("_".join(line.split("\t")[0].split("_")[:-1]))
    f.close()

    fileStr = ""
    for dirpath, dirnames, filenames in os.walk(CLIPS_DIR_PATH):
        for f in filenames:
            if f in files:
                [Fs, x] = audioBasicIO.readAudioFile(dirpath + "/" + f);
                features = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
                fileStr = fileStr + f + "\t" + np.array_str(features) + "\n"
                print fileStr
                return

extractFeatures()