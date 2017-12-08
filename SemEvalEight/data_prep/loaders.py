# File Author: Morgan Stuart
import json
from os import listdir, path
from SemEvalEight import config
import numpy as np
import nltk

from nltk.corpus import wordnet

# From SO: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'


# Sourced from code provided in Task (generateData), with some aspects modified
def generate_subtask1_data(file_indices=None, tokenized_folder=None, verbose=False,
                           lowercase=True, lemmatize=True):
    """
    Generator form load_subtask1_data
    """
    sentence = ''
    relevance = 0
    if tokenized_folder is None:
        tokenized_folder = config.tokenized_dir
    print("Loading tokens from file: %s" % tokenized_folder)
    files = listdir(tokenized_folder)

    print("Found %d token files" % len(files))
    print(file_indices)
    lemmatizer = nltk.wordnet.WordNetLemmatizer()
    for i, fileName in enumerate(files):
        if file_indices is not None and i not in file_indices:
            continue
        if verbose:
            print("Loading %s" % fileName)

        tk_path = path.join(tokenized_folder, fileName)
        with open(tk_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '\n' in line and line.strip() == '':
                    if sentence != '':
                        if lowercase:
                            sentence = sentence.lower()

                        if lemmatize:
                            _sent = " ".join([lemmatizer.lemmatize(wrd, pos=get_wordnet_pos(pos))
                                              for wrd, pos in nltk.pos_tag(nltk.word_tokenize(sentence))])
                        else:
                            _sent = sentence


                        yield _sent, relevance
                    sentence = ''
                    relevance = 0
                else:
                    if sentence == '':
                        sentence = line.split(' ')[0]
                    else:
                        sentence += ' ' + line.split(' ')[0]
                    if line[:-1].split(' ')[-1] != 'O':
                        relevance = 1


def generate_subtask2_data(file_indices=None, tokenized_folder=None,
                           target_encoding_method='basic', verbose=False):
    if tokenized_folder is None:
        tokenized_folder = config.tokenized_dir
    print("Loading tokens from file: %s" % tokenized_folder)
    files = listdir(tokenized_folder)

    print("Found %d token files" % len(files))
    print(file_indices)

    if target_encoding_method == 'basic':
        # Encode begin and others as the same
        # Creating 4 classes (one extra class for no label?)
        target_enc_map = {
            'B-Entity': np.array([1, 0, 0, 0]),
            'I-Entity': np.array([1, 0, 0, 0]),

            'B-Action': np.array([0, 1, 0, 0]),
            'I-Action': np.array([0, 1, 0, 0]),

            'B-Modifier': np.array([0, 0, 1, 0]),
            'I-Modifier': np.array([0, 0, 1, 0])
        }
        no_label_target_code = np.array([0, 0, 0, 1])

    elif target_encoding_method == 'detail':
        target_enc_map = {
            'B-Entity': np.array([1, 0, 0, 0, 0, 0, 0]),
            'I-Entity': np.array([0, 1, 0, 0, 0, 0, 0]),

            'B-Action': np.array([0, 0, 1, 0, 0, 0, 0]),
            'I-Action': np.array([0, 0, 0, 1, 0, 0, 0]),

            'B-Modifier': np.array([0, 0, 0, 0, 1, 0, 0]),
            'I-Modifier': np.array([0, 0, 0, 0, 0, 1, 0])
        }
        no_label_target_code = np.array([0, 0, 0, 0, 0, 0, 1])

    skip_chars = {'•'}

    for i, fileName in enumerate(files):
        if file_indices is not None and i not in file_indices:
            continue
        if verbose:
            print("Loading %s" % fileName)

        tk_path = path.join(tokenized_folder, fileName)

        with open(tk_path, 'r', encoding='utf-8') as f:

            sentence = list()
            word_pos = 0
            for line in f:
                if any(sc in line for sc in skip_chars):
                    continue

                ln = line.strip()

                if ln == '':
                    yield sentence
                    sentence = list()
                    word_pos = 0
                    continue

                try:
                    word, mdb_label = ln.split()
                except:
                    print("Bad line: %s" % ln)
                    print(fileName)
                    print(line)
                    #raise

                if mdb_label == 'O':
                    begin_ind = tk_label = None
                    target_enc = no_label_target_code
                else:
                    begin_ind, tk_label = mdb_label.split('-')
                    target_enc = target_enc_map[mdb_label]


                tk_info = dict(token=word,
                               #pos=pos,
                               pos=None,
                               original_word_position=word_pos,
                               file_origin=fileName,
                               is_begin=begin_ind,
                               label = tk_label,
                               target_vector=target_enc)

                sentence.append(tk_info)
                word_pos += 1


def load_subtask2_data(file_indices, tokenized_folder=None):
    return list(generate_subtask2_data(file_indices=file_indices,
                                       tokenized_folder=tokenized_folder))


def load_subtask1_data(file_indices, tokenized_folder=None, sampe_frac=.7):
    """
    Loads raw X, Y into memory, where X samples are senteces and Y samples are integers with
    Y == 1 indicating a relevant sentence (binary classification).

    :param file_indices: set of file indexes to load (index into os.listdir list)
    :param tokenized_folder: Path to folder containing *.tokens files (usually 'tokenized')
    :return: X, Y as tuple(<list>, <list>)
    Where X is a list of sentences and Y is a list of integers where 1 means the sentence is relevant
    """
    X = list()
    Y = list()
    for x, y in generate_subtask1_data(file_indices, tokenized_folder=tokenized_folder):
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


def load_subtask1_brown_auto_labeled(top_n=10):
    auto_label_p =  'top_%d_auto_labeled_from_brown_external_att.txt' % top_n
    p = path.join(config.ext_data_dir, auto_label_p)

    X_auto = open(p, encoding='utf-8').readlines()
    Y_auto = np.ones(len(X_auto))
    return X_auto, Y_auto

def load_auto_labeled(top_n=10):
    auto_label_p =  'top_%d_auto_labeled_from_additional_plaintext.txt' % top_n
    p = path.join(config.ext_data_dir, auto_label_p)

    X_auto = open(p, encoding='utf-8').readlines()
    Y_auto = np.ones(len(X_auto))
    return X_auto, Y_auto


def load_stucco_annotations(full_corpus_path=None):
    if full_corpus_path is None:
        if not hasattr(config, 'stucco_corpus_json_path'):
            raise ValueError("Stucco full corpus path is None and has not been set globally")
        else:
            full_corpus_path = config.stucco_corpus_json_path

    with open(full_corpus_path, 'r') as f:
        corpus_json = json.load(f)
    return corpus_json


def load_glove_wiki_embedding(word_index=None,
                              glove_embedding_path=None,
                              embedding_dim=100,
                              max_words=1000000,
                              add_one_for_mask=True,
                              return_embedding_map=False):
    if glove_embedding_path is None:
        glove_embedding_path = config.embeddings_dir

    file_p = path.join(glove_embedding_path,
                       'glove.6B.%dd.txt' % embedding_dim)
    f = open(file_p, encoding='utf-8')

    embeddings_index = dict()
    for idx, line in enumerate(f):
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except ValueError as ve:
            print("ERRRROR on line %d: %s\r" % (idx, line[:20]))
            continue

        embeddings_index[word] = coefs
    f.close()

    if return_embedding_map:
        return embeddings_index
    elif word_index is None:
        raise ValueError("Must pass word index if an embedding matrix is being built")

    # prepare embedding matrix
    nb_words = min(max_words, max(word_index.values()) + 1)
    if add_one_for_mask: nb_words += 1

    # words not found in embedding index will be all-zeros.
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    num_missing = 0
    for word, i in word_index.items():
        if i >= nb_words:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            num_missing += 1

    print("Loaded embedding of shape %s" % str(embedding_matrix.shape))
    print("Number of words missing %d (%f%%)" % (num_missing, (num_missing*100.)/len(word_index)))
    return embedding_matrix


def load_word2vec_embedding(embed_name, word_index=None, max_words=1000000, embedding_dim=200,
                            add_one_for_mask=True, return_embedding_map=False):
    from gensim.models.keyedvectors import KeyedVectors

    embed_name_map = {'general_with_all_cyber':'combotext-vector.bin',
                         'all_cyber':'papertext2-vector.bin',
                         'labeled_cyber': 'papertext-vector.bin',
                         'general': 'text8-vector.bin'}
    embed_name = embed_name_map[embed_name]
    embedding_dir = config.embeddings_dir

    embed_path = path.join(embedding_dir, embed_name)
    model = KeyedVectors.load_word2vec_format(embed_path,
                                              binary=True,
                                              unicode_errors='ignore')

    embedding_index = {wrd: model.syn0[idx] for idx, wrd in enumerate(model.index2word)}
    if return_embedding_map:
        return embedding_index

    assert word_index is not None
    nb_words = min(max_words, max(word_index.values()) + 1)
    if add_one_for_mask: nb_words += 1
    embedding_matrix = np.zeros((nb_words, embedding_dim))


    num_missing = 0
    for word, i in word_index.items():
        if i >= nb_words:
            continue

        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            num_missing += 1

    print("Loaded embedding of shape %s" % str(embedding_matrix.shape))
    print("Number of words missing %d (%f%%)" % (num_missing, (num_missing * 100.)/len(word_index)))

    return embedding_matrix

    #return model.return_keras_embedding()

if __name__ == """__main__""":

    print("Testing load functions")
    #print("Loading subtask 1")
    X, Y = load_subtask1_data(list(range(10)))#,
                              #tokenized_folder='/home/morgan/ownCloud/Classes/NLP/semeval_task_8/data/tokenized/')
    #print("\tDone")
    #print("loading word2vec embedding")
#    auto_labeled = load_word2vec_embedding('general_with_all_cyber')
    #print("Done")

    print("loading auto labeled")
    auto_labeled = load_auto_labeled(5)
    print("Done: %d elements" % len(auto_labeled))

    print("Loading subtask 2")
    t = list(generate_subtask2_data([0]))
    print("\tDone")

    print("Loading stucco data")
    stucco = load_stucco_annotations()
    print("\tDone")
