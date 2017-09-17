import os

# Hardcode data dirs if wanting to avoid env vars
semeval8_data_dir = None

if 'SEMEVAL_8_DATA' in os.environ:
    semeval8_data_dir = os.environ['SEMEVAL_8_DATA']


if semeval8_data_dir is None:
    raise ValueError("Specify semeval data dir in config.py or set the 'SEMEVAL_8_DATA' environment variable")

tokenized_dir = os.path.join(semeval8_data_dir, 'tokenized')