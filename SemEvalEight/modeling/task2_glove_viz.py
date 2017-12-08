# Author: Fatima
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from os import listdir, path
from sklearn.decomposition import PCA
from SemEvalEight import config


"""

Goes through all the annotated files and return a list of all the sentences filter by the label_type parameter.

:param label_type: The three possible token 'Entity' - 'Action' - 'Modifier'
:return: List of sentences. 
"""
def get_token_labels(label_type):
    if (label_type != 'Entity' and label_type != 'Action' and label_type != 'Modifier'):
        print('Erroorr, invalid label_type')
        return

    tokenized_folder = config.semeval8_data_dir + '/annotations'
    sentence = list()

    print("Loading tokens from file: " + tokenized_folder + " - with label_type: " + label_type)
    files = listdir(tokenized_folder)

    print("Found %d token files" % len(files))

    for i, fileName in enumerate(files):
        if fileName.endswith('ann'):
            # print("Loading %s" % fileName)

            tk_path = path.join(tokenized_folder, fileName)

            with open(tk_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line == '\n':
                        continue

                    if line[0] == 'T':
                        _, token_label, words = line.split('\t')

                        token_label = token_label.split(' ', 1)[0]
                        if (token_label == 'Subject' or token_label == 'Object'):
                            token_label = 'Entity'

                        if (label_type == token_label):
                            sentence.append(words.rstrip('\n').lower())
    return (sentence)


def load_glove(glove_file_name):
    vectors = {}
    with open(config.embeddings_dir + '/' + glove_file_name) as f:
        for line in tqdm(f, total=400000):
            word, vector = line.split(maxsplit=1)
            v = np.fromstring(vector, sep=' ', dtype='float32')
            vectors[word] = v / np.linalg.norm(v)
    return vectors


plt.rcParams["figure.figsize"] = (18, 10)


"""

We use the glove pre-trained data with the labeled tokens and apply dimensionality reduction.

:param words: List of labeled tokens.
:return: array-like, shape (n_samples, n_components) 
"""
def get_vector_words(*words):
    pca = PCA(n_components=2)

    final_words = list()

    for wo in words:
        try:
            if vectors[wo] is not None:
                final_words.append(wo)
        except KeyError:
            # print('ERROR: key not found! -> ' + wo)
            continue

    #print("final words len: " + str(len(final_words)))
    xys = pca.fit_transform([vectors[w] for w in final_words])

    return xys

    #for word, xy in zip(final_words, xys):
        #plt.annotate(word, xy, fontsize=20)


if __name__ == """__main__""":
    glove_file_name = 'glove.42B.300d.txt'
    #glove_file_name = 'glove.twitter.27B.50d.txt'
    #glove_file_name = 'glove.twitter.27B.100d.txt'
    #glove_file_name = 'glove.twitter.27B.200d.txt'
    #glove_file_name = 'glove.twitter.27B.25d.txt'


    # Create plots of Glove embeddings for task2
    # (embeddings for labeled tokens -> PCA(n=2)-> scatter plot of the three labels)

    vectors = load_glove(glove_file_name)

    sent_entity = get_token_labels('Entity')
    sent_action = get_token_labels('Action')
    sent_modifier = get_token_labels('Modifier')

    print(len(sent_entity))
    print(len(sent_action))
    print(len(sent_modifier))

    xys_entity = get_vector_words(*sent_entity)
    xys_action = get_vector_words(*sent_action)
    xys_modifier = get_vector_words(*sent_modifier)

    plt.scatter(*xys_entity.T, color='navy', label='Entity', s=75, alpha=0.5)
    plt.scatter(*xys_action.T, color='turquoise', label='Action', s=75, alpha=0.5)
    plt.scatter(*xys_modifier.T, color='darkorange', label='Modifier', s=75, alpha=0.5)

    plt.legend(fontsize=18)
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.show()