#!/usr/bin/env bash
# Author: Morgan Stuart

TASK_8_PATH=$(pwd)
pushd .

cd $TASK_8_PATH/SemEvalEight/modeling

source ./venv/bin/activate

# FROM PROJECT DOCS:
# All projects must be run from the command line using a bash shell script named runit.sh – this script
# should compile your program/s and run them. Any file names or other parameter settings must be made
# in this script and not in your code. Do not hard code file or directory names in source code files.  
export THEANO_FLAGS="mode=FAST_RUN,device=gpu,floatX=float32"

# REQUIRED: Locations of the data
export SEMEVAL_8_DATA="$TASK_8_PATH/data"
export STUCCO_AUTO_LABELED="$TASK_8_PATH/ext_data/stucco_corpus"
export EMBEDDINGS_DIR="$TASK_8_PATH/ext_data/embeddings"


echo "Training ML-Models on unigram and bi-gram bag-of-words"
echo "-->Running original train set"
python task1_bag_of_words.py --eval-best
echo "-->Running original+auto-labeled train set"
python task1_bag_of_words.py --eval-best --top-n=10


echo "Training GloVe Embedding + LSTM classifier (eta. 40-60 minutes with GPU)"
python task1_embedding.py --depth 7 \
                          --hidden-size=20 \
                          --embed-dim=100 \
                          --learning-rate=0.00000017

deactivate
popd