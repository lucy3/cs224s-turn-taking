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

TIME_PATH = '/afs/ir/data/linguistic-data/Switchboard/SWBD-MSState-Transcripts/swb_ms98_transcriptions'
ANNOTATED_PATH = './swda/'
FILLERS = ['you see', 'you know', 'anyways', 'actually', 'yeah', 'okay', 'anyway', 'now', 'ok', 'like', 'well', 'say', 'so', 'see', 'uh', 'um', 'oh', 'huh', 'uhhuh', 'huhuh', 'right', 'sure', 'yes', 'yep', 'hum', 'umhum', 'hm', 'thats right', 'thats true', 'true', 'wow', 'nice', 'cool']

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

def label(turns, backchannel, words, times):
    """
    TODO everything here
    TODO add output file as an input parameter
    Inputs:
        - turns: list of SENTENCES for each turn
        - backchannel: same length as turns, list of booleans
        - words: list of WORDS 
        - times: same length as words, list of end-start tuples
    """
    tok_turns = ' '.join(turns).split() # list of WORDS
    """
    # TODO: 
    should get all indices that are at the end of a turn in tok_turns
    To do this I would count the lengths of each turn in turns that is split
    so the first index would be the length of the first tokenized turn, 
    the second index would be the added length of the first and second, 
    the third index would be the added length of the first three tokenized
    turns... and so on. 
    Similarily, one should get indices in tok_turns that are associated
    with backchannels. 
    """
    seq = SequenceMatcher()
    seq.set_seqs(tok_turns, words) # take in two lists of words for each transcription
    # TODO: remove this print statement. It is here for now so you can see how
    # the alignment is operating. 
    for tag, i1, i2, j1, j2 in seq.get_opcodes():
        print ("%7s a[%d:%d] (%s) b[%d:%d] (%s)" % (tag, i1, i2, tok_turns[i1:i2], j1, j2, words[j1:j2]))
    '''
    tag - edit operation, which is replace, insert, delete, equal
    i1, i2 - indices of tok_turns
    j1, j2 - corresponding indices of words
    Non problematic edit operations:
        replacing slices of the same length. treat those indices as equal. 
        equal. treat those indices as equal. 
    Other edit operations are problematic.
    Somehow store a mapping between indices in words and indices in tok_turns!
    If the indices are problematic ones, maybe store the mapping as idx -> None.

    For every filler in words: 
        BE CAREFUL OF FILLERS THAT ARE TWO WORDS LONG. In those cases you
        want to remember the previous word. Check for 'you see' before
        you check for 'see', etc. The start and end times of those will 
        be the start of the first word and the end of the second word. 
        BE CAREFUL OF IF-STATEMENT ORDERING. We want to make sure
        we label reinforcements first since they are often single words
        that would be labeled as starts/ends if we did a different ordering. 
        If the filler is in a problematic index in the variable words, 
            ignore it and continue.
        If the filler is mapped to an index that is associated
        with backchanneling, label it as reinforcement. 
        If the filler's mapped index in tok_turns is after an index that
            is the end of turn, it is labeled as seeking a turn. 
        If a filler is mapped to an index in tok_turns that is at
            the end of turn, it is labeled as end of a turn. 
        Otherwise, it is labeled as a continuation. 
    Don't forget to output these labels into the output file so we
    don't have to rerun label collection when we're trying to collect
    audio clips... 
    Print often!! scream!!
    '''

def align(pairs):
    """
    This function calls the other functions
    that reads in the data into workable forms. 
    Note that a_turns is a list of strings for each turn, while
    A_words is a list of words. 
    """
    # TODO: open an output file to write to
    # That tab delimited output file should have one column that is well-designed
    # ID for a filler. I suggest dialogueID_starttime_endtime_lexical,
    # e.g. 4726_1.0298_1.6098_uh 
    # the second column would be the label for that filler. 
    # This output file would later be used to extract clips. 
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
        label(a_turns, a_backchannel, A_words, A_times)
        label(b_turns, b_backchannel, B_words, B_times)
        i += 1 # TODO: remove this break when done testing on small subset of data.
        if i > 5:
            break

def main():
    pairs = pair_files()
    align(pairs)    

main()
