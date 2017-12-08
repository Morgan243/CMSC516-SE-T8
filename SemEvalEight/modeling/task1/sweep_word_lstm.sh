#!/usr/bin/env bash


for((word_depth=1;word_depth < 5; word_depth = word_depth + 1))
do
    for((word_width=16;word_width < 257; word_width = word_width + 16))
    do
        time python task1_character_level_lstm.py --embed-dim=50 \
                                                  --word-maxlen=50 \
                                                  --dense-dropout=.5 \
                                                  --dense-depth=3 \
                                                  --dense-size=128 \
                                                  --learning-rate=1. \
                                                  --word-depth=$word_depth \
                                                  --word-hidden-size=$word_width \
                                                  --no-char \
                                                  --word-dropout=.3 \
                                                  --append-to='test_results.json'
    done
done