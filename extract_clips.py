import numpy as np
import collections

def extract():
#open labels file
    labelsDict = defaultDict(list)
    with open(labels) as f:
        for line in f:
            fileInfo = line.split("\t")
            firstInfoSplit = fileInfo[0].split("_")
            key = firstInfoSplit[0] + firstInfoSplit[1]
            timeTuple = (firstInfoSplit[2], firstInfoSplit[3])
            labelsDict[key].append(timeTuple)


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
#open audio file using py sox, extract clip and save it, 
