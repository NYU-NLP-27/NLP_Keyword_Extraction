# Data imports
import pandas as pd
from datasets import load_dataset
import statistics

# Page Rank imports
import nltk
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
import networkx as nx

dataset = load_dataset("midas/inspec", "raw")
data = dataset['test']

df = data.data.to_pandas()

# POS tag words, find synset of word and create new list

synset_documents = [] # list of the list of synsets for each document
for doc in df['document']:
    doc_sets = [] # for this document, the list of synsets

    for word in doc:
        #print(word)
        pos_tag = nltk.pos_tag([word])[0][1]
        #doc_tags.append(pos_tag)
        #print(pos_tag)
        
        if pos_tag.__contains__('NN'):
            part = wn.NOUN
        elif pos_tag.__contains__('V'):
            part = wn.VERB  
        elif pos_tag.__contains__('JJ'):
            part = wn.ADJ
        elif pos_tag.__contains__('RB'):
            part = wn.ADV
        else:
            continue
        synset = wn.synsets(word, pos=part)
        #print(synset)
        
        if len(synset) > 0:
            doc_sets.append(synset[0]) # append the synset for this word

    synset_documents.append(doc_sets)


key_words = []
for doc in range(len(synset_documents[:100])):
    G = nx.Graph()

    for i in range(len(synset_documents[doc])):
        for j in range(len(synset_documents[doc])):
            one = synset_documents[doc][i]
            two = synset_documents[doc][j]
            G.add_edge(one, two, weight=one.path_similarity(two))


    rank = nx.pagerank(G)

    kwords = list(rank.keys())[:10]

    for i in range(len(kwords)):
        kwords[i] = kwords[i].name().split(".")[0]

    key_words.append(kwords)


hit_rate = []
precision = []
recall = []
fone = []

for kword in range(len(key_words)):
    if len(df['extractive_keyphrases'][kword]) == 0:
        continue

    answers = " ".join(df['extractive_keyphrases'][kword])

    hr = 0

    for word in key_words[kword]:
        if word in answers:
            hr += 1
    
    pr = hr / len(key_words[kword])
    rc = hr / len(df['extractive_keyphrases'][kword])
    if pr == 0 or rc == 0:
        f1 = 0
    else:
        f1 = 2/((1/pr) + (1/rc))

    hit_rate.append(hr)
    precision.append(pr)
    recall.append(rc)
    fone.append(f1)


#print(hit_rate)
#print(precision)
#print(recall)
#print(fone)

print(statistics.fmean(hit_rate))
print(statistics.fmean(precision))
print(statistics.fmean(recall))
print(statistics.fmean(fone))
