"""
Labels filler words as one of the following: 
- continuation
- ceding the floor
- seeking the floor
- reinforcement
"""

from swda import Transcript
import os
import re
import string
from collections import defaultdict

TIME_PATH = '/afs/ir/data/linguistic-data/Switchboard/SWBD-MSState-Transcripts/swb_ms98_transcriptions'
ANNOTATED_PATH = './swda/'
FILLERS = ['you see', 'you know', 'anyways', 'actually', 'yeah', 'okay', 'anyway', 'now', 'ok', 'like', 'well', 'say', 'so', 'see', 'uh', 'um', 'oh', 'huh', 'uh-huh', 'huh-uh', 'right', 'sure', 'yes', 'yep', 'um-hum', 'hm', 'that\'s right', 'that\'s true', 'true', 'wow', 'nice', 'cool']

def pair_files():
    pairs = defaultdict(list) # IDs to pair [] 
    for dirpath, dirnames, filenames in os.walk(ANNOTATED_PATH):
        for f in filenames:
            if '.DS_Store' not in f and 'swda-metadata' not in f:
                file_path = dirpath + '/' + f
                ID = re.findall('([0-9]+)\.utt\.csv', file_path)[0]
                pairs[ID].append(file_path)
    for dirpath, dirnames, filenames in os.walk(TIME_PATH):
        for f in filenames:
            if 'AAREADME' not in f and 'sw-ms98-dict.text' not in f and 'word' in f:
                file_path = dirpath + '/' + f
                ID = file_path.split('/')[-2] 
                if ID in pairs:
                    pairs[ID].append(file_path)
    for ID in pairs:
        assert len(pairs[ID]) == 3
    return pairs

def get_turns(ann_file):
    a_backchannel = [] # array of booleans
    a_turns = [] # each element is a turn 
    b_backchannel = []
    b_turns = []
    trans = Transcript(ann_file, './swda/swda-metadata.csv')
    prev_line = -1
    a_curr_turn = ''
    b_curr_turn = ''
    a_ended = False
    b_ended = False
    a_act = None
    b_act = None
    reinf_tags = set(['b', 'b^r', 'bh', 'bk'])
    for utt in trans.utterances:
        current_line = utt.utterance_index
        if current_line != prev_line: 
            if utt.caller == 'A' and a_ended == True:
                a_turns.append(a_curr_turn)
                a_backchannel.append(a_act)
                a_curr_turn = ''
            elif utt.caller == 'B' and b_ended == True:
                b_turns.append(b_curr_turn)
                b_backchannel.append(b_act)
                b_curr_turn = ''
        if utt.caller == 'A':
            if utt.text.strip()[-1] == '/':
                a_ended = True
            else:
                a_ended = False
        if utt.caller == 'B':
            if utt.text.strip()[-1] == '/':
                b_ended = True
            else:
                b_ended = False
        clean_test = utt.text.replace('{D', '').replace('{F','').replace('{E', '').replace('{C','').replace('{A','')
        clean_test = re.sub('<[a-zA-Z\s]*>', '', clean_test)
        clean_test = clean_test.lower().translate(None, string.punctuation)
        clean_test = re.sub(' +',' ', clean_test).strip()
        if utt.caller == 'A':
            a_act = utt.act_tag in reinf_tags
            a_curr_turn += ' ' + clean_test
        if utt.caller == 'B':
            b_act = utt.act_tag in reinf_tags
            b_curr_turn += ' ' + clean_test
        prev_line = current_line
    a_turns.append(a_curr_turn)
    b_turns.append(b_curr_turn)
    a_backchannel.append(a_act)
    b_backchannel.append(b_act)
    return (a_turns, b_turns, a_backchannel, b_backchannel)

def get_times(A_file, B_file):
    A_words, A_times, B_words, B_times = [],[],[],[]
    return (A_words, A_times, B_words, B_times)

def align(pairs):
    for ID in pairs:
        p = pairs[ID]
        A_time_file = None
        B_time_file = None
        ann_file = p[0]
        if p[1].split('-')[-4].endswith('A'): 
            A_time_file = p[1]
            B_time_file = p[2]
        else:
            A_time_file = p[2]
            B_time_file = p[1]
        a_turns, b_turns, a_backchannel, b_backchannel = get_turns(ann_file)
        A_words, A_times, B_words, B_times = get_times(A_time_file, B_time_file)
        break
    # REMINDER: remove apostrophes from time data since we removed punctuation in annotated data

def main():
    pairs = pair_files()
    align(pairs)    

main()
