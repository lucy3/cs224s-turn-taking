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
from difflib import SequenceMatcher 

TIME_PATH = '/afs/ir/data/linguistic-data/Switchboard/SWBD-MSState-Transcripts/swb_ms98_transcriptions'
ANNOTATED_PATH = './swda/'
FILLERS = ['you see', 'you know', 'anyways', 'actually', 'yeah', 'okay', 'anyway', 'now', 'ok', 'like', 'well', 'say', 'so', 'see', 'uh', 'um', 'oh', 'huh', 'uhhuh', 'huhuh', 'right', 'sure', 'yes', 'yep', 'hum', 'umhum', 'hm', 'thats right', 'thats true', 'true', 'wow', 'nice', 'cool']

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
                a_turns.append(a_curr_turn.strip())
                a_backchannel.append(a_act)
                a_curr_turn = ''
            elif utt.caller == 'B' and b_ended == True:
                b_turns.append(b_curr_turn.strip())
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
        clean_test = re.sub('<[_a-zA-Z\s]*>', '', clean_test)
        clean_test = clean_test.lower().translate(None, string.punctuation)
        clean_test = re.sub(' +',' ', clean_test).strip()
        if utt.caller == 'A':
            a_act = utt.act_tag in reinf_tags
            a_curr_turn += ' ' + clean_test
        if utt.caller == 'B':
            b_act = utt.act_tag in reinf_tags
            b_curr_turn += ' ' + clean_test
        prev_line = current_line
    a_turns.append(a_curr_turn.strip())
    b_turns.append(b_curr_turn.strip())
    a_backchannel.append(a_act)
    b_backchannel.append(b_act)
    return (a_turns, b_turns, a_backchannel, b_backchannel)

def get_times(A_file, B_file):
    A_words, A_times, B_words, B_times = [],[],[],[]
    with open(A_file, 'r') as f_A:
        for line in f_A: 
            items = line.split()
            if not items[3].startswith('['):
                A_times.append((float(items[1]), float(items[2])))
                word = items[3].lower().strip().translate(None, string.punctuation+string.digits)
                if word == 'umhum': word = 'uhhuh' # spelling consistency
                A_words.append(word)
    with open(B_file, 'r') as f_B:
        for line in f_B:
            items = line.split()
            if not items[3].startswith('['):
                B_times.append((float(items[1]), float(items[2])))
                word = items[3].lower().strip().translate(None, string.punctuation+string.digits)
                if word == 'umhum': word = 'uhhuh'
                B_words.append(word) 
    return (A_words, A_times, B_words, B_times)

def align(pairs):
    i = 0
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
        # right now this code is only doing things for speaker A
        # so that I can test things to see if they work
        turns = ' '.join(a_turns).split()
        # should get all indices that are at the end of a turn in the list of turns
        # also should get indices in this list associated with backchannels
        seq = SequenceMatcher()
        seq.set_seqs(turns, A_words)
        for tag, i1, i2, j1, j2 in seq.get_opcodes():
            if tag != 'equal': 
                print ("%7s a[%d:%d] (%s) b[%d:%d] (%s)" % (tag, i1, i2, turns[i1:i2], j1, j2, A_words[j1:j2]))
        # if replace:
            # if the replaced slice is the same length, treat those indices as equal
        # if equal: 
            # treat those indices as equal
        # other edit ops are labeled as problematic
        # for every filler in A_words:
            # if filler is in problematic index, ignore
            # if filler is after its equivalent index in turns that is associated with an end of turn, 
            # it is labeled as a seeking a turn
            # if a filler is at its equivalent index in turns associated with end of turn, it is a end
            # otherwise it is a continuation 
            # if a filler is at its equivalent index in turns associated with a backchannel, it is a reinforcement
        i += 1
        if i > 5:
            break

def main():
    pairs = pair_files()
    align(pairs)    

main()
