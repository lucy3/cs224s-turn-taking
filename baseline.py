"""
Gathers filler words.

Ideally I would want something like a disfluency annotated data alongside
the word-timed data but unfortunately I only have a sample of disfluency data.
I used that sample to find what words are fillers (F) or discourse markers (D).

Some words such as "and," "so," "like," and "man" are both fillers/discourse
markers as well as other kinds of words.

When we get the full Switchboard transcript it would be nice to use that to
determine a turn. For now, I define a continuation as one where
the other speaker stays silent after a filler and an ending as one where
the other speaker is or starts talking.
"""
import string
import re
import os
from collections import Counter

DATADIR = "./swb_ms98_transcriptions/"
DFS = "./switchboard_sample/disfluency"

FILLERS = set()

def gather_fillers():
    """
    Get filler words and print some basic stats about
    these conversations.
    """
    first = []
    last = []
    prev = []
    with open(DFS, 'r') as file:
        for line in file:
            matches = re.findall(r'\{D ([A-Za-z \#\.\,]*)? \}', line)
            matches.extend(re.findall(r'\{F ([A-Za-z \#\.\,]*)? \}', line))
            FILLERS.update([m.lower().translate(None, string.punctuation) for m in matches])
            cleaned = line.lower().translate(None, string.punctuation).split()
            if cleaned and re.match('[ab][0-9]+', cleaned[0]):
                if len(cleaned) > 1:
                    first.append(cleaned[1])
                if len(prev) > 1:
                    last.append(prev[-1])
            prev = cleaned
    print "Most common turn starters"
    print Counter(first).most_common(20)
    print "Most common turn enders"
    print Counter(last).most_common(20)
    print "Words marked as F and D"
    print FILLERS

def get_data(dialogue):
    """
    Turn word transcriptions into something workable.
    """
    A_word = []
    B_word = []
    for i in range(len(dialogue)):
        f = dialogue[i]
        f_name = f.split('-')
        if f_name[0].endswith('A') and f_name[-1] == 'word.text':
            with open(f, 'r') as f_A:
                for line in f_A:
                    A_word.append(tuple(line.split()[1:]))
        elif f_name[0].endswith('B') and f_name[-1] == 'word.text':
            with open(f, 'r') as f_B:
                for line in f_B:
                    B_word.append(tuple(line.split()[1:]))
    return A_word, B_word

def extract_simple_feats(dialogue): 
    """
    Turn data into feature vectors of what filler
    it has, length of that filler, length of silence
    immediately after it. Silences end when someone (either
    the current speaker or the other speaker) talk.
    """
    features = [] # list of feature vectors for each example
    labels = [] # correct label
    A_word, B_word = get_data(dialogue)
    # watch out for two-word fillers like "you know"
    # TODO
    return features, labels

def main():
    gather_fillers()
    dialogues = []
    for dirpath, dirnames, filenames in os.walk(DATADIR):
        if filenames and '.DS_Store' not in filenames:
            dialogues.append([dirpath + '/' + f for f in filenames])

    # implementing on one set of dialogue for now
    dialogues[0]
    features, labels = extract_simple_feats(dialogues[0])


if __name__ == '__main__':
    main()