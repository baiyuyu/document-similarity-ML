from nltk.tokenize import word_tokenize
import numpy as np
import string
import pandas as pd

# corpus of four documents
docs = ["Sachin is considered to be one of the greatest cricket players",
        "Federer is considered one of the greatest tennis players",
        "Nadal is considered one of the greatest tennis players",
        "Virat is the captain of the  Indian cricket team"

        ]


def createVocab(docList):
        vocab = {}
        for doc in docList:
                print(doc)
                doc = doc.translate(str.maketrans('', '', string.punctuation))

                words = word_tokenize(doc.lower())
                for word in words:
                        if (word in vocab.keys()):
                                vocab[word] = vocab[word] + 1
                        else:
                                vocab[word] = 1
        return vocab


vocab = createVocab(docs)

# Compute document term matrix as well idf for each term

termDict = {}

docsTFMat = np.zeros((len(docs), len(vocab)))

docsIdfMat = np.zeros((len(vocab), len(docs)))

docTermDf = pd.DataFrame(docsTFMat, columns=sorted(vocab.keys()))
docCount = 0
for doc in docs:
        doc = doc.translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(doc.lower())
        for word in words:
                if (word in vocab.keys()):
                        docTermDf[word][docCount] = docTermDf[word][docCount] + 1

        docCount = docCount + 1
        

#Computed idf for each word in vocab
idfDict={}

for column in docTermDf.columns:
    idfDict[column]= np.log((len(docs) +1 )/(1+ (docTermDf[column] != 0).sum()))+1
    