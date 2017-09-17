from os import listdir, path


# Sourced from code provided in Task (generateData), with some aspects modified
def generate_subtask1_data(file_indices, tokenized_folder):
    """
    Generator form load_subtask1_data
    """
    sentence = ''
    relevance = 0
    print("Loading tokens from file: %s" % tokenized_folder)
    files = listdir(tokenized_folder)

    print("Found %d token files" % len(files))
    print(file_indices)

    for i, fileName in enumerate(files):
        if i not in file_indices:
            continue
        print("Loading %s" % fileName)

        tk_path = path.join(tokenized_folder, fileName)
        with open(tk_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    if sentence != '':
                        yield sentence, relevance
                    sentence = ''
                    relevance = 0
                else:
                    if sentence == '':
                        sentence = line.split(' ')[0]
                    else:
                        sentence += ' ' + line.split(' ')[0]
                    if line[:-1].split(' ')[-1] != 'O':
                        relevance = 1


def load_subtask1_data(file_indices, tokenized_folder):
    """
    Loads raw X, Y into memory, where X samples are senteces and Y samples are integeres with
    Y == 1 indicating a relevant sentence.

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
    return X, Y


if __name__ == """__main__""":
    X, Y = load_subtask1_data(list(range(10)),
                              tokenized_folder='/home/morgan/ownCloud/Classes/NLP/semeval_task_8/data/tokenized/')
    print(X)
    print(Y)
    print("done")