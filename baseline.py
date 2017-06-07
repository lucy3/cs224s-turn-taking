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
the other speaker is or starts talking. (So, hold the floor or support/cede
the floor.)
"""
import string
import numpy as np
import re
import os
from collections import Counter
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

DATADIR = "./swb_ms98_transcriptions/"
DFS = "./switchboard/disfluency"
OUTPUT = "./baseline_features"
FILLERS = set(['actually', 'anyway', 'anyways', 'cool', 'hm', 'huh', 'huhuh', 'hum', 'like', 'nice', 'now', 'oh', 'okay', 'right', 'say', 'see', 'so', 'sure', 'thats right', 'thats true', 'true', 'uh', 'uhhuh', 'um', 'well', 'wow', 'yeah', 'yep', 'yes', 'you know', 'you see', 'uh-hum'])

def gather_fillers():
    """
    Get filler words and print some basic stats about
    these conversations.
    """
    first = []
    last = []
    prev = []
    fillers_set = set()
    last_tot = 0
    first_tot = 0
    with open(DFS, 'r') as file:
        for line in file:
            matches = re.findall(r'\{D ([A-Za-z \#\.\,]*)? \}', line)
            matches.extend(re.findall(r'\{F ([A-Za-z \#\.\,]*)? \}', line))
            fillers_set.update([m.lower().translate(None, string.punctuation) for m in matches])
            cleaned = line.lower().translate(None, string.punctuation).split()
            if cleaned and re.match('[ab][0-9]+', cleaned[0]):
                if len(cleaned) > 1:
                    first_tot += 1
                    if cleaned[1] == 'f' or cleaned[1] == 'd' or cleaned[1] == 'c':
                        first.append(cleaned[2])
                    else:
                        first.append(cleaned[1])
                if len(prev) > 1:
                    last_tot += 1
                    last.append(prev[-1])
            prev = cleaned
    print "Total starters, enders"
    print first_tot, last_tot
    print "Most common turn starters"
    print Counter(first).most_common(20)
    print "Most common turn enders"
    print Counter(last).most_common(20)
    print "Words marked as F and D"
    if 'you  know' in fillers_set:
        fillers_set.remove('you  know')
    if 'you known' in fillers_set:
        fillers_set.remove('you known')
    if 'un' in fillers_set:
        fillers_set.remove('un')
    fillers = list(fillers_set)
    print fillers
    return fillers

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
                    items = line.split()
                    A_word.append((float(items[1]), float(items[2]), items[3]))
        elif f_name[0].endswith('B') and f_name[-1] == 'word.text':
            with open(f, 'r') as f_B:
                for line in f_B:
                    items = line.split()
                    B_word.append((float(items[1]), float(items[2]), items[3]))
    return A_word, B_word

def get_time_dict(A_word, B_word):
    A_time = {k :[] for k in range(int(A_word[-1][0]) + 1)}
    for item in A_word:
        time_bucket = int(item[0])
        A_time[time_bucket].append(item)
    B_time = {k :[] for k in range(int(B_word[-1][0]) + 1)}
    for item in B_word:
        time_bucket = int(item[0])
        B_time[time_bucket].append(item)
    return A_time, B_time

def get_pause_length(end_time, time1, time2):
    """
    inputs:
    - end_time: float representing end_time of filler
    - time1: dictionary of time buckets for speaker 1
        speaker 1 is the person who said the filler
        and is either continuing or ending
    - time2: dictionary of time buckets for speaker 2
    returns:
    - pause duration, and a label that is 0 if it's
    a continuation, 1 if it is an end
    """
    bucket = int(end_time)
    if bucket > 1:
        bucket -= 1
    while bucket in time1 or bucket in time2:
        if bucket in time1:
            candidates1 = time1[bucket]
            for cand in candidates1:
                if cand[0] <= end_time and cand[1] > end_time and \
                        cand[2] != '[silence]' and cand[2] != '[noise]':
                    return 0, 0
                if cand[0] > end_time and cand[2] != '[silence]' and \
                        cand[2] != '[noise]':
                    return cand[0] - end_time, 0
        if bucket in time2:
            candidates2 = time2[bucket]
            for cand in candidates2:
                if cand[0] <= end_time and cand[1] > end_time and \
                        cand[2] != '[silence]' and cand[2] != '[noise]':
                    return 0, 1
                if cand[0] > end_time and cand[2] != '[silence]' and \
                        cand[2] != '[noise]':
                    return cand[0] - end_time, 1
        bucket += 1
    return 0, 1    # end of dialogue

def extract_simple_feats_helper(words1, fillers,
        time1, time2, features, labels):
    prev = None
    nxt = None
    for i in range(len(words1)):
        item = words1[i]
        if i > 0:
            prev = words1[i - 1]
        if i < len(words1) - 1:
            nxt = words1[i + 1]
        vec = [0] * (len(fillers) + 2)
        time_start = item[0]
        time_end = item[1]
        word = item[2]
        if prev is not None and prev[2] + ' ' + word in fillers:
            vec[fillers.index(prev[2] + ' ' + word)] = 1
            vec[-2] = time_end - float(prev[0])
            # print item
            if nxt is None:
                lab = 1
                vec[-1] = 0
            elif nxt[2] != '[silence]' and nxt[2] != '[noise]':
                lab = 0
                vec[-1] = 0
            else:
                vec[-1], lab = get_pause_length(time_end, time1, time2)
            # print "pause length", vec[-1], lab
            labels.append(lab)
            features.append(vec)
        elif word in fillers:
            vec[fillers.index(word)] = 1
            vec[-2] = time_end - time_start
            # print item
            if nxt is None:
                lab = 1
                vec[-1] = 0
            elif nxt[2] != '[silence]' and nxt[2] != '[noise]':
                lab = 0
                vec[-1] = 0
            else:
                vec[-1], lab = get_pause_length(time_end, time1, time2)
            # print "pause length", vec[-1], lab
            labels.append(lab)
            features.append(vec)
    return features, labels

def extract_simple_feats(dialogue, fillers): 
    """
    Turn data into feature vectors of what filler
    it has and the length of the pause immediately after it.
    Pauses end when someone (either
    the current speaker or the other speaker) talk.
    """
    features = []   # list of feature vectors for each example
    labels = []     # correct label. 0 if continue, 1 if end.
    A_word, B_word = get_data(dialogue)
    A_time, B_time = get_time_dict(A_word, B_word)
    features, labels = extract_simple_feats_helper(A_word, fillers,
        A_time, B_time, features, labels)
    features, labels = extract_simple_feats_helper(B_word, fillers,
        B_time, A_time, features, labels)
    assert len(labels) == len(features)
    return features, labels

def do_data_stuff(features, labels):
    dialogues = []
    for dirpath, dirnames, filenames in os.walk(DATADIR):
        if filenames and '.DS_Store' not in filenames:
            dialogues.append([dirpath + '/' + f for f in filenames])

    for i in range(len(dialogues)):
        print "extracting features for", dialogues[i]
        f, l = extract_simple_feats(dialogues[i], FILLERS)
        features.extend(f)
        labels.extend(l)

    out = open(OUTPUT, 'w')
    for i in range(len(labels)):
        line = str(labels[i]) + ' ' + ' '.join([str(f) for f in features[i]]) + '\n'
        out.write(line)
    return features, labels

def main():
    features, labels = [], []
    if not os.path.isfile(OUTPUT):
        features, labels = do_data_stuff(features, labels)
    else:
        with open(OUTPUT, 'r') as features_labels:
            for line in features_labels:
                items = line.split()
                labels.append(float(items[0]))
                features.append([float(i) for i in items[1:]])

    print len(features), len(labels)
    contin = []
    end = []
    for i in range(len(labels)):
        if labels[i] == 0:
            contin.append(features[i][-1])
        elif labels[i] == 1:
            end.append(features[i][-1])
    print "continuation pause average", sum(contin)/len(contin)
    print "continuation std", np.std(contin)
    print "b/t turns pause average", sum(end)/len(end)
    print "b/t turns std", np.std(end)

    words = list(FILLERS)

    features = np.array(features, dtype=float)
    labels = np.array(labels, dtype=float)
    newFeatures = [features[i] for i in range(len(labels)) if labels[i] == 1.0]
    newLabels = [labels[i] for i in range(len(labels)) if labels[i] == 1.0]
    targetCount = len(newLabels)
    count = 0
    for i in range(len(labels)):
        if labels[i] == 0.0:
            newFeatures.append(features[i])
            newLabels.append(labels[i])
            count = count + 1
            if count == targetCount:
                break

    newFeatures = np.array(newFeatures)
    newLabels = np.array(newLabels)

    print "SGD"
    print "All"
    sgdClassifier = linear_model.SGDClassifier()
    sgdClassifier.fit(newFeatures, newLabels)
    print("weights")
    weights = sgdClassifier.coef_
    for i in range(len(words)):
        print words[i] + ": " + str(weights[0][i])

    print("Weight for duration of the word: " + str(weights[0][-2]))
    print("Weight for duration of the pause: " + str(weights[0][-1]))

    print("scores")
    scores = cross_val_score(sgdClassifier, newFeatures, newLabels)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print "Words only"

    sgdClassifier1 = linear_model.SGDClassifier()
    sgdClassifier1.fit(newFeatures[:, [i for i in range(len(words))]], newLabels)
    print("weights")
    weights = sgdClassifier1.coef_
    for i in range(len(words)):
        print words[i] + ": " + str(weights[0][i])

    #print("Weight for duration of the word: " + str(weights[0][-2]))
    #print("Weight for duration of the pause: " + str(weights[0][-1]))

    print("scores")
    scores = cross_val_score(sgdClassifier1, newFeatures[:, [i for i in range(len(words))]], newLabels)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print "Duration of the word and the pause"

    sgdClassifier2 = linear_model.SGDClassifier()
    sgdClassifier2.fit(newFeatures[:, [-2, -1]], newLabels)
    print("weights")
    weights = sgdClassifier2.coef_
    print("Weight for duration of the word: " + str(weights[0][-2]))
    print("Weight for duration of the pause: " + str(weights[0][-1]))

    print("scores")
    scores = cross_val_score(sgdClassifier2, newFeatures[:, [-2, -1]], newLabels)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print "Duration of the pause"

    sgdClassifier3 = linear_model.SGDClassifier()
    sgdClassifier3.fit(newFeatures[:, [-1]], newLabels)
    print("weights")
    weights = sgdClassifier3.coef_
    print("Weight for duration of the pause: " + str(weights[0][-1]))

    print("scores")
    scores = cross_val_score(sgdClassifier3, newFeatures[:, [-1]], newLabels)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print "Random forest classifier"
    print "All"
    clf = RandomForestClassifier()
    clf = clf.fit(newFeatures, newLabels)
    weights = clf.feature_importances_
    for i in range(len(words)):
        print words[i] + ": " + str(weights[i])

    print("Importance for duration of the word: " + str(weights[-2]))
    print("Importance for duration of the pause: " + str(weights[-1]))
    scores = cross_val_score(clf, newFeatures, newLabels)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print "Words only"
    clf1 = RandomForestClassifier()
    clf1 = clf1.fit(newFeatures[:, [i for i in range(len(words))]], newLabels)
    weights = clf1.feature_importances_
    for i in range(len(words)):
        print words[i] + ": " + str(weights[i])

    #print("Importance for duration of the word: " + str(weights[-2]))
    #print("Importance for duration of the pause: " + str(weights[-1]))
    scores = cross_val_score(clf1, newFeatures[:, [i for i in range(len(words))]], newLabels)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print "Duration of words and pauses"
    clf2 = RandomForestClassifier()
    clf2 = clf2.fit(newFeatures[:, [-2, -1]], newLabels)
    weights = clf2.feature_importances_

    print("Importance for duration of the word: " + str(weights[-2]))
    print("Importance for duration of the pause: " + str(weights[-1]))
    scores = cross_val_score(clf2, newFeatures[:, [-2, -1]], newLabels)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print "Duration of pauses"
    clf3 = RandomForestClassifier()
    clf3 = clf3.fit(newFeatures[:, [-1]], newLabels)
    weights = clf3.feature_importances_

    print("Importance for duration of the pause: " + str(weights[-1]))
    scores = cross_val_score(clf3, newFeatures[:, [-1]], newLabels)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':
    main()