import numpy as np
import collections
import os
import re
import sox

AUDIO_DIR_PATH = '../Audio/Audio-swbd1ph2/' 

def extract():
#open labels file
    labelsDict = collections.defaultdict(list)
    with open('./labels') as f:
        for line in f:
            fileInfo = line.split("\t")
            firstInfoSplit = fileInfo[0].split("_")
            key = firstInfoSplit[0] + firstInfoSplit[1]
            timeTuple = (firstInfoSplit[2], firstInfoSplit[3])
            labelsDict[key].append(timeTuple)

    for dirpath, dirnames, filenames in os.walk(AUDIO_DIR_PATH):
        for f in filenames:
            if '.sph' in f:
                ID = re.findall('([0-9]+)', f)[0][1:]

                if ID + "A" in labelsDict.keys() and ID + "B" in labelsDict.keys():
                    for timestamps in labelsDict[ID + "A"]:
                        tfm = sox.Transformer()
                        tfm.remix({1: [1], 2: [2]})
                        tfm.trim(float(timestamps[0]), float(timestamps[1]))
                        print(dirpath + '/' + f)
                        check = bool(os.path.dirname('./clips/' + ID + '_A_' + timestamps[0] + '_' + timestamps[1] + '.sph')) 
                        check1 = not os.access(os.getcwd(), os.W_OK)
                        check2 = not os.access(os.path.dirname('./clips/' + ID + '_A_' + timestamps[0] + '_' + timestamps[1] + '.sph'), os.W_OK)
                        if check:
                            print "done1"
                        if check1:
                            print "done2"
                        if check2:
                            print "done3"

                        output = tfm.build(dirpath + '/' + f, 'C:/clips/' + ID + '_A_' + timestamps[0] + '_' + timestamps[1] + '.sph', return_output=True)
                        print output
                        return

extract()
"""
#read line by line
#split by tab...length 2 (id_blahblah  label) 

#split id_blalbh by _ DIALOG ID (sw#dialogID), speaker, start - 1, end + 1
#import pysox..extract length of audioclip
#lexical value


#dictionary (dialog id + speaker-> rest of chunk)

#read shit from albel file





def collectAudioFiles():

#os.walk
#split each clip into left and right channels (sox remix1 remix 2 os.system___command line shit)
#open audio file using py sox, extract clip and save it
"""