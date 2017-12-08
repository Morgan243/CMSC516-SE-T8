from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import nltk
import numpy as np
import pandas as pd
import argparse

from SemEvalEight.data_prep.loaders import load_subtask1_data, get_wordnet_pos
from SemEvalEight.config import tokenized_dir, brown_ext_dir, ext_data_dir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk.tokenize import sent_tokenize
import nltk
from glob import glob

def build_train_count_vectorizer(raw_data=None):
    if raw_data is None:
        raw_data, _ = load_subtask1_data(list(range(53)))

    cvec = CountVectorizer(ngram_range=(1, 2),
                           stop_words='english',
                           min_df=3)

    return cvec.fit(raw_data)


def fit_committee(models=None, feature_type='bow'):
    if models is None:
        models = dict(
            nb=MultinomialNB(alpha=3.25),
            gb=GradientBoostingClassifier(n_estimators=170, max_depth=5,
                                                    learning_rate=0.5,
                                                    min_samples_leaf=3, min_samples_split=4),
                      dt=DecisionTreeClassifier(criterion='gini', max_depth=25,
                                                max_leaf_nodes=None, min_samples_leaf=3,
                                                min_samples_split=4))

    raw_X, Y = load_subtask1_data(list(range(53)))
    if feature_type == 'bow':
        cvec = CountVectorizer(ngram_range=(1, 2),
                               stop_words='english',
                               min_df=3)

        X = cvec.fit_transform(raw_X).toarray()

    vote_clf = VotingClassifier(estimators=[(k, v) for k, v in models.items()], n_jobs=3,
                                flatten_transform=False,
                                voting='soft').fit(X, Y)

    return vote_clf


def load_raw_from_sentence_file(glob_path=os.path.join(brown_ext_dir, '*.txt')):
    att_files = glob(glob_path)
    num_files = len(att_files)
    if num_files == 0:
        raise ValueError("No ATT files found - make sure you are using the older dataset to use this")
    print("Found %d files" % num_files)

    unlabeld_seq_dict = {os.path.split(file_p)[-1]
                         :[" ".join(nltk.word_tokenize(ln.strip()))  for ln in open(file_p, 'r', encoding='utf-8').readlines()]
                            for file_p in att_files}
    return unlabeld_seq_dict


def load_raw_from_text(glob_path=os.path.join(brown_ext_dir, '*.txt')):
    txt_files = glob(glob_path)
    print("Checking %s" % glob_path)
    print("Found %d files" % len(txt_files))

    unlabeld_seq_dict = {os.path.split(file_p)[-1]:open(file_p, 'r').read()
                            for file_p in txt_files}
    blacklist = set(['<', '>', 'image:'])
    clean_seq_map = dict()
    lemmatizer = nltk.wordnet.WordNetLemmatizer()

    for name, raw_txt in unlabeld_seq_dict.items():
        sequences = sent_tokenize(raw_txt)
        tok_seq = [[lemmatizer.lemmatize(wrd, pos=get_wordnet_pos(pos))
                    for wrd, pos in nltk.pos_tag(nltk.word_tokenize(sent))]
                          for sent in sequences]

        #tok_seq = [nltk.word_tokenize(s)
        #           for s in sequences if not any(bl in s
        #                                         for bl in blacklist)]
        clean_seq = [" ".join(s) for s in tok_seq]
        clean_seq_map[name] = np.array(clean_seq)

    return clean_seq_map


def auto_label_provided_external(top_n_per_doc=3):
    # prelabeled setences by them?
    #unlabeled_seq_map = load_raw_from_sentence_file()

    unlabeled_seq_map = load_raw_from_text()

    vectorizer = build_train_count_vectorizer(raw_data=None)

    bow_seq = {k:vectorizer.transform(v).toarray()
               for k, v in unlabeled_seq_map.items()}

    cmt = fit_committee()

    # Todo: Pick only top n from each document - supposing that each document has something to say
    labeled = dict()
    for k, samples in bow_seq.items():
        raw_samples = unlabeled_seq_map[k]
        clf_probas = cmt.transform(samples)
        sample_avg_proba = clf_probas[:, :, 1].mean(axis=0)

        preds_df = pd.DataFrame(dict(proba=sample_avg_proba, txt=raw_samples))
        top_n_raw = preds_df.sort_values('proba', ascending=False).txt.iloc[:top_n_per_doc].values
        #print(top_n_raw)
        #print(k)
        labeled[k] = top_n_raw
        #cmt.predict

    #print(labeled)
    return labeled




if __name__ == """__main__""":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name",
                        help="Output file to write lines labeled positive",
                        default=None,
                        type=str)
    parser.add_argument('--top-n', type=int,
                        default=5,
                        help='Take the top-n highest scoring from each document')
    args = parser.parse_args()
    print("Running top-n of %d" % args.top_n)


    labeled = auto_label_provided_external(top_n_per_doc=args.top_n)
    if args.output_name is None:
        fname = 'top_%d_auto_labeled_from_brown_external_att.txt'%args.top_n
        output_p = os.path.join(ext_data_dir,
                                fname)
    else:
        output_p = args.output_name

    with open(output_p, 'w', encoding='utf-8') as f:
        for _, sentences in labeled.items():
            #f.writelines(sentences)
            for s in sentences:
                if '\n' in s:
                    f.write(s)
                else:
                    f.write(s + '\n')
