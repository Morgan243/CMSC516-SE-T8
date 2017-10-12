# Author: Morgan Stuart
import time
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import sys
import uuid
import argparse

from keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from keras import layers as kl
import keras

import numpy as np
import pandas as pd

from SemEvalEight.data_prep.loaders import load_subtask1_data, load_subtask1_brown_auto_labeled
from SemEvalEight.data_prep import loaders
from SemEvalEight.data_prep import preprocessing
from SemEvalEight.utils import binary_classification_metrics

def build_keras_embedding_classifier(embeddings, activation='tanh',
                                     optimizer='adam',
                                     hidden_size=20, depth=2,
                                     lr=0.001, decay=0.003,
                                     dropout=.5, recurrent_dropout=.33):
    em_shape = embeddings.shape
    num_vectors = em_shape[0]
    vector_dim = em_shape[1]

    emb = kl.Embedding(num_vectors, vector_dim,
                             weights=[embeddings],
                       input_length=None,
                             trainable=False)

    m = keras.models.Sequential()
    m.add(emb)
    for d in range(depth):
        m.add(kl.Dropout(dropout))
        m.add(kl.LSTM(hidden_size, activation=activation,
                      recurrent_dropout=recurrent_dropout,
                      return_sequences= d != (depth - 1)))

    m.add(kl.Dropout(dropout))
    m.add(kl.Dense(1, activation='sigmoid'))
    keras.optimizers.Adam(lr=lr, decay=decay)
    m.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    print(m.summary())
    return m


def build_params_from_grid(activations=['tanh'],
                           hidden_size=[20, 30], depth=[3, 4],#range(1, 3),
                           lr=[0.0000003], decay=[0.001],
                           dropout=[.5], recurrent_dropout=[.33]):
    return [
        dict(activation=a,
             hidden_size=hs,
             depth=d, lr=_lr,
             decay=dy, dropout=drp,
             recurrent_dropout=rd)

        for a in activations
        for hs in hidden_size
        for d in depth
        for _lr in lr
        for dy in decay
        for drp in dropout
        for rd in recurrent_dropout
    ]


def grid_search(embeddings,
                train_X, train_y,
                cv_X, cv_y,
                grid_params,
                ):

    results = list()
    err = False
    total_p = len(grid_params)
    for i, params in enumerate(grid_params):
        print("Running [%d/%d]:\n%s" % (i+1, total_p,
                                        str(params)))
        m = build_keras_embedding_classifier(embeddings, **params)

        cb = [EarlyStopping(patience=17),
              TerminateOnNaN(),
              ReduceLROnPlateau(verbose=1)]

        m.fit(train_X, train_y, epochs=100,
              batch_size=128, validation_data=(cv_X, cv_y),
              callbacks=cb)

        test_preds = m.predict(cv_X).round().astype(int)
        test_actual = Y_test

        pred = m.predict(np.concatenate([sequences, test_sequences])).round().astype(int)
        actual = np.concatenate([Y, Y_test])

        try:
            perf_metrics = binary_classification_metrics(actual, pred)
            test_metrics = binary_classification_metrics(test_actual, test_preds)
            test_metrics = {"test_%s" % k : v for k, v in test_metrics.items()}
            print("HOLDOUT PERFORMANCE: %s" % str(test_metrics))
        except ValueError as ve:
            print("VALUE ERROR")
            perf_metrics = dict()
            err=True
        m_id = str(uuid.uuid4())[:7]
        m.save('./saved_models/model_%s.keras'%m_id)
        results.append(dict(error=err, model_id=m_id,
                            **params, **perf_metrics, **test_metrics))

    # embedding distance - embed from different sources with same seed of important words, say from Glove. then tune all others around thos
    # can then compare them.

    return results


def load_data(embedding_dim=100, return_holdout=False):
    file_ixs = list(range(39))
    X, Y = load_subtask1_data(file_ixs[:23])

    # Add in auto labeled
    X_auto, Y_auto = load_subtask1_brown_auto_labeled()
    X = np.concatenate([X, X_auto])
    Y = np.concatenate([Y, Y_auto])


    ix = list(range(len(X)))
    np.random.shuffle(ix)
    X = X[ix]
    Y = Y[ix]


    X_test, Y_test = load_subtask1_data(file_ixs[23:31])
    test_ix = list(range(len(X_test)))
    np.random.shuffle(test_ix)

    X_test = X_test[test_ix]
    Y_test = Y_test[test_ix]

    tokenizer, sequences = preprocessing.tokenize_texts(X, nb_words=3000)
    # TODO: Masking implementation rather than padding?
    sequences = pad_sequences(sequences, maxlen=100)

    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_sequences = pad_sequences(test_sequences, maxlen=100)

    embeddings = loaders.load_glove_wiki_embedding(tokenizer.word_index,
                                                   embedding_dim=embedding_dim)

    if return_holdout:
        X_holdout, Y_holdout = load_subtask1_data(file_ixs[31:])
        holdout_sequences = tokenizer.texts_to_sequences(X_holdout)
        holdout_sequences = pad_sequences(holdout_sequences, maxlen=100)
        return embeddings, sequences, Y, test_sequences, Y_test, holdout_sequences, Y_holdout
    else:
        return embeddings, sequences, Y, test_sequences, Y_test


if __name__ == """__main__""":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-search',
                        default=False,
                        action='store_true')
    parser.add_argument('--model-load',
                        default=None,
                        type=str)
    parser.add_argument('--dropout',
                        default=0.5,
                        type=float)
    parser.add_argument('--depth',
                        default=5,
                        type=int)
    parser.add_argument('--hidden-size',
                        default=20,
                        type=int)
    parser.add_argument('--learning-rate',
                        default=2.5e-7,
                        type=float)
    parser.add_argument('--recurrent-dropout',
                        default=.5,
                        type=float)
    parser.add_argument('--decay',
                        default=.0035,
                        type=float)
    parser.add_argument('--embed-dim',
                        default=100,
                        type=int)

    parser.add_argument('--model-output-path',
                        default='auto')
    parser.add_argument('--metrics-output-path',
                        default='auto')

    args = parser.parse_args()

    ret = load_data(embedding_dim=args.embed_dim, return_holdout=True)
    embeddings, sequences, Y, test_sequences, Y_test, holdout_sequences, Y_holdout = ret

    if args.model_load is not None:
        print("Keras model loading broken!")
        sys.exit(1)

        print("Loading model at %s" % str(args.model_load))
        m = keras.models.load_model(args.model_load)
        print("Running predictions on datasets")
        train_pred = m.predict(sequences)
        dev_pred = m.predict(test_sequences)
        holdout_pred = m.predict(holdout_sequences)

        train_metrics = binary_classification_metrics(Y, train_pred.round())
        dev_metrics = binary_classification_metrics(Y_test, dev_pred.round())
        holdout_metrics = binary_classification_metrics(Y_holdout, holdout_pred.round())

        print("Train: %s" % str(train_metrics))
        print("Dev: %s" % str(dev_metrics))
        print("Holdout: %s" % str(holdout_metrics))

    elif args.grid_search:
        grid_params = build_params_from_grid(activations=['tanh'],
                                            hidden_size=[15, 20, 25], depth=[4, 5, 6],#range(1, 3),
                                            lr=[0.0000002], dropout=[.5],
                                             decay=[0.0035],
                                             recurrent_dropout=[0.5])

        res = grid_search(embeddings=embeddings,
                          train_X=sequences, train_y=Y,
                          cv_X=test_sequences, cv_y=Y_test,
                          grid_params=grid_params)

        res_df = pd.DataFrame(res)

        name = 'task_1_embedding_lstm_grid_search_res_%d.csv' % int(time.time())
        p = os.path.join(args.metrics_output_dir, name )
        res_df.to_csv(p)

    else:

        uid = uuid.uuid4()
        cb = [EarlyStopping(patience=17),
              TerminateOnNaN(),
              ReduceLROnPlateau(verbose=1)]
        m = build_keras_embedding_classifier(embeddings=embeddings,
                                             lr=args.learning_rate, depth=args.depth,
                                             hidden_size=args.hidden_size,
                                             #lr=2.5e-7, depth=5, hidden_size=20,
                                             decay=args.decay,
                                             recurrent_dropout=args.recurrent_dropout)

        hist = m.fit(sequences, Y, epochs=100, batch_size=128,
                      validation_data=(test_sequences, Y_test),
                      callbacks=cb)


        pred = m.predict(test_sequences).round().astype(int)
        metrics = binary_classification_metrics(Y_test, pred)
        print("Dev perf")
        print(metrics)

        holdout_pred = m.predict(holdout_sequences).round().astype(int)
        holdout_metrics = binary_classification_metrics(Y_holdout, holdout_pred)
        print("Holdout perf")
        print(holdout_metrics)


        hist_df = pd.DataFrame(hist.history)

        if args.metrics_output_path == 'auto':
            p = 'task1_embedding_%s.csv' % str(uid)
        else:
            p = args.metrics_output_path
        hist_df.to_csv(p)

        #if args.model_output_path == 'auto':
        #    m.save("model_%s.keras" % str(uid))
        #else:
        #    m.save(args.model_output_path)
