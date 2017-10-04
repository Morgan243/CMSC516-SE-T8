import os

# Todo: setup a real plaintext config?

###----
# Hardcode data dirs if wanting to avoid env vars
semeval8_data_dir = ''
stucco_corpus_dir = ''
embeddings_dir = ''

###----
# From env vars
semeval8_data_dir = os.environ.get('SEMEVAL_8_DATA', semeval8_data_dir)
stucco_corpus_dir = os.environ.get('STUCCO_AUTO_LABELED', stucco_corpus_dir)
embeddings_dir = os.environ.get('EMBEDDINGS_DIR', embeddings_dir)

###----
# Check and warn
if semeval8_data_dir is None:
    raise ValueError("Specify semeval data dir in config.py or set the 'SEMEVAL_8_DATA' environment variable")

# not a required data set
if stucco_corpus_dir is None:
    print("warning: Specify stucco data dir in config.py or set the 'STUCCO_AUTO_LABELED' environment variable")
    print("download stucco auto-labeled here: https://github.com/stucco/auto-labeled-corpus")
    #raise ValueError("Specify stucco data dir in config.py or set the 'STUCCO_AUTO_LABELED' environment variable")
else:
    stucco_corpus_json_path = os.path.join(stucco_corpus_dir, 'full_corpus.json')

if embeddings_dir is None:
    print("warning: specify embeddings dir in config.py or set the 'EMBEDDINGS_DIR' environment variable")

# Useful paths
tokenized_dir = os.path.join(semeval8_data_dir, 'tokenized')
