import os

### BEGIN CONFIG ###
TOKENIZED_DATA_DIR="data/tokenized/"

### END CONFIG ###

fileList = [f for f in os.listdir(TOKENIZED_DATA_DIR) if os.path.isfile(os.path.join(TOKENIZED_DATA_DIR, f))]
#print(fileList)

numTokens = 0
numRelevantTokens = 0
numFiles = 0
numSentences = 0
numRelevantSentences = 0
for fileName in fileList:
    numFiles = numFiles + 1
    textFile = open(TOKENIZED_DATA_DIR+fileName)
    lastLineWasEmpty = 0
    relevantTokensInSentence = 0
    for line in textFile:
        strippedLine = line.strip()
        if strippedLine != '':
            lastLineWasEmpty = 0
            numTokens = numTokens + 1
            lineComponents = strippedLine.split()
            if lineComponents[-1].strip() != 'O':
                numRelevantTokens = numRelevantTokens + 1
                relevantTokensInSentence = relevantTokensInSentence+1
        else:
            if lastLineWasEmpty == 0:
                numSentences = numSentences+1
                if relevantTokensInSentence != 0:
                    numRelevantSentences = numRelevantSentences+1
            lastLineWasEmpty = 1
            relevantTokensInSentence = 0

    textFile.close()

print("====================")
print("Basic stats")
print("====================")
print("Number of files examined: " + str(numFiles))
print("Total number of tokens: " + str(numTokens))
print("Total number of sentences: " + str(numSentences))
print("Average number of tokens per file: " + str(numTokens/numFiles))
print("Average number of sentences per file: " + str(numSentences/numFiles))

print("")
print("====================")
print("Task 1 stats")
print("====================")
print("Number of tokens relevant to Task 1: " + str(numRelevantTokens))
print("Number of sentences relevant to Task 1: " + str(numRelevantSentences))
print("Percentage of relevant tokens: " + str(numRelevantTokens/numTokens))
print("Percentage of relevant sentences: " + str(numRelevantSentences/numSentences))
print("Average number of relevant tokens per file: " + str(numRelevantTokens/numFiles))
print("Average number of relevant sentences per file: " + str(numRelevantSentences/numFiles))




