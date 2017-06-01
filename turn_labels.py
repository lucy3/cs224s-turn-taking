"""
Labels filler words as one of the following: 
- continuation
- ceding the floor
- seeking the floor
- reinforcement

You want to download Chris Pott's swda.py file as
well as swda.zip and put it in the directory that this 
file is in. 
To clone a repo on corn, go to /farmshare/user_data/your_SUNetID
instead of cloning it on afs. 
"""

from swda import Transcript
import os
import re
import string
from collections import defaultdict
from difflib import SequenceMatcher 

TIME_PATH = './swb_ms98_transcriptions/' 
#TIME_PATH = '/afs/ir/data/linguistic-data/Switchboard/SWBD-MSState-Transcripts/swb_ms98_transcriptions/'
ANNOTATED_PATH = './swda/'
ONE_FILLERS = ['anyways', 'actually', 'yeah', 'okay', 'anyway', 'now', 'ok', 'like', 'well', 'say', 'so', 'see', 'uh', 'um', 'oh', 'huh', 'uhhuh', 'huhuh', 'right', 'sure', 'yes', 'yep', 'hum', 'umhum', 'hm', 'true', 'wow', 'nice', 'cool']
TWO_FILLERS = ['you see', 'you know', 'thats right', 'thats true']
OUTPUT = './labels'

def pair_files():
    """
    This function pairs the data files together. 
    Returns: 
        - pairs: a dictionary of dialogue IDs to three files, the first
        of which is the annotated disfluency transcript and then two
        which are the time-stamped word transcripts. This leads to a total
        of three files per dialogue. 
    """
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
    """
    This very long function takes in an disfluency annotated file
    and returns several lists. 
    Returns:
    - a_backchannel: boolean of true/false corresponding to whether
    the utterance, such as 'uh-huh', is labeled as a backchannel using
    the utterance tag provided by swda. This list is of the same
    length as a_turns. 
    - a_turns: a list of strings, where each element is a turn. 
    I carefully calculated turns so that parts that 
    overlapped with a reinforcement was considered a turn. 
    These strings are stripped of punctuation, random noises
    such as coughing, and lowercase. 
    - b_backchannel and b_turns are implemented similarily. 
    Apologies for long, winded code. 
    """
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
        clean_test = re.sub('<[_a-zA-Z/\s]*>', '', clean_test)
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
    """
    Note that I respelled umhum as uhhuh. There are also
    cases where um and uh are different between the two transcripts
    but I did not change that because that inconsistency is less
    consistent than this inconsistency... if you know what I mean.
    This function takes in two time-stamped word transcripts, one
    for speaker A and one for speaker B. 
    Returns:
        - A_words: a list of words for speaker A
        - A_times: a list of the same length as A_words that has
        tuples of start and end times for each word. 
        - B_words and B_times are similar. 
        Note that punctuation is removed and the words are lowercase. 
    """
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

def label(turns, backchannel, words, times, out, ID, speaker):
    """
    Inputs:
        - turns: list of SENTENCES for each turn
        - backchannel: same length as turns, list of booleans
        - words: list of WORDS 
        - times: same length as words, list of start-end tuples
    """
    turn_indices = set()
    pointer = 0
    turn_indices.add(-1)
    reinf_indices = set()
    for i in range(len(turns)): 
        turn = turns[i]
        if backchannel[i] == True:
            reinf_indices.update(range(pointer, pointer + len(turn.split())))
        pointer += len(turn.split())
        turn_indices.add(pointer-1)
    tok_turns = ' '.join(turns).split() # list of WORDS
    print ID, speaker
    print ' '.join(turns)
    print ' '.join(words)
    print
    seq = SequenceMatcher()
    seq.set_seqs(tok_turns, words) # take in two lists of words for each transcription
    time_to_turns = {i:None for i in range(len(words))}
    # Only align same-length replacements or equivalent components 
    for tag, i1, i2, j1, j2 in seq.get_opcodes():
        if (tag == 'replace' and i2-i1 == j2-j1) or tag == 'equal':
            match = i1
            for j in range(j1, j2):
                time_to_turns[j] = match
                match += 1
    for i in range(len(words)):
        w = words[i]
        if not time_to_turns[i]:
            continue
        start, end = times[i]
        label = None
        # first, two word fillers: 
        if ((w == 'know' or w == 'see') and i - 1 > 0 and words[i-1] == 'you') or\
            ((w == 'true' or w == 'right') and i - 1 > 0 and words[i-1] == 'thats'):
                start, _ = times[i-1]
                if time_to_turns[i-1] and time_to_turns[i-1] not in turn_indices: 
                    # this two-word phrase is part of the same turn
                    if time_to_turns[i] in reinf_indices: 
                        label = 'reinforce'
                    elif time_to_turns[i-1]-1 in turn_indices:
                        label = 'seek'
                    elif time_to_turns[i] in turn_indices:
                        label = 'end'
                    else:
                        label = 'continue'
                    out.write(ID + '_' + speaker + '_' + str(start) + '_' + \
                            str(end) + '_' + words[i-1] + w + '\t' + label + '\n')
        elif w in ONE_FILLERS: # then, one word fillers
            if time_to_turns[i] in reinf_indices:
                label = 'reinforce'
            elif time_to_turns[i]-1 in turn_indices:
                label = 'seek'
            elif time_to_turns[i] in turn_indices:
                label = 'end'
            else:
                label = 'continue'
            out.write(ID + '_' + speaker + '_' + str(start) + '_' + \
                    str(end) + '_' + w + '\t' + label + '\n')

def align(pairs):
    """
    This function calls the other functions
    that reads in the data into workable forms. 
    Note that a_turns is a list of strings for each turn, while
    A_words is a list of words. 
    """
    out = open(OUTPUT, 'w')
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
        label(a_turns, a_backchannel, A_words, A_times, out, ID, 'A')
        label(b_turns, b_backchannel, B_words, B_times, out, ID, 'B')

def main():
    pairs = pair_files()
    align(pairs)    

main()
