STUCCO_URL='https://github.com/stucco/auto-labeled-corpus/archive/master.zip'
wget $STUCCO_URL
unzip master.zip
mv auto-labeled-corpus-master/corpus/ ./stucco_corpus
rm master.zip
rm -rf auto-labeled-corpus-master

GLOVE_6B_URL='http://nlp.stanford.edu/data/glove.6B.zip'
wget $GLOVE_6B_URL
unzip glove.6B.zip
mkdir embeddings
rm glove.6B.zip
mv glove.*.txt embeddings

