from __future__ import unicode_literals
import numpy as np
import json
import math
import spacy
from numpy import dot
from numpy.linalg import norm
nlp = spacy.load('en_core_web_md')


# COLOUR VECTORS


colour_data = json.loads(open("colours.json").read())

# Converts colors from hex format (#1a2b3c) to a tuple of integers
def hex_to_int(s):
    s = s.lstrip("#")
    return int(s[:2], 16), int(s[2:4], 16), int(s[4:6], 16)

# Creates a dictionary and populates it with mappings from color names to RGB vectors for each color in the data
colors = dict()
for item in colour_data['colors']:
    colors[item["color"]] = hex_to_int(item["hex"])

# Calculates the Euclidean distance between two points
def distance(coord1, coord2):
    v1 = np.array(coord1)
    v2 = np.array(coord2)
    return np.linalg.norm(v1 - v2)

# Subtracts one vector from another
def subtractv(coord1, coord2):
    v1 = np.array(coord1)
    v2 = np.array(coord2)
    return v1 - v2

# Adds two vectors together
def addv(coord1, coord2):
    v1 = np.array(coord1)
    v2 = np.array(coord2)
    return v1 - v2

# Calculates the mean from a list of vectors
def meanv(coords):
    a = np.array(coords)
    return np.mean(a, axis=0)

# Testing the json file has been successfully read in
'''print(colors['olive'])
print(colors['red'])
print(colors['black'])
print(distance([10, 1], [5, 2]))
print(subtractv([10, 1], [5, 2]))
print(meanv([[0, 1], [2, 2], [4, 3]]))'''

def closest(space, coord, n=20):
    closest = []
    for key in sorted(space.keys(), key=lambda x: distance(coord, space[x]))[:n]:
        closest.append(key)
        # Adds the vector value of the colour
        closest.append(space[key])
    return closest

#print(closest(colors, colors['red']))
# The vector resulting from subtracting "red" from "purple," we get a series of "blue" colors
#print(closest(colors, subtractv(colors['purple'], colors['red'])))

doc = nlp(open("test.txt").read())
# use word.lower_ to normalize case
drac_colors = [colors[word.lower_] for word in doc if word.lower_ in colors]
avg_color = meanv(drac_colors)
#print(closest(colors, avg_color))


# WORD VECTORS IN SPACY


# All of the words in the text file
tokens = list(set([w.text for w in doc if w.is_alpha]))

# Gets the vector of a given string from spaCy's vocabulary:
def vec(s):
    return nlp.vocab[s].vector

# Cosine similarity
def cosine(v1, v2):
    if norm(v1) > 0 and norm(v2) > 0:
        return dot(v1, v2) / (norm(v1) * norm(v2))
    else:
        return 0.0
    
# Iterates through a list of tokens and returns the token whose vector is most similar to a given vector
def spacy_closest(token_list, vec_to_check, n=10):
    return sorted(token_list,
                  key=lambda x: cosine(vec_to_check, vec(x)),
                  reverse=True)[:n]

#Shows the cosine similarity between dog and puppy is larger than the similarity between trousers and octopus
print(cosine(vec('dog'), vec('puppy')) > cosine(vec('trousers'), vec('octopus')))

# What's the closest equivalent of basketball?
print(spacy_closest(tokens, vec("basketball")))

# Halfway between day and night
print(spacy_closest(tokens, meanv([vec("day"), vec("night")])))

# What would we get if we added these two together?
print(spacy_closest(tokens, addv(vec("water"), vec("frozen"))))


# SENTENCE SIMILARITY IN SPACY


# To get the vector for a sentence, we simply average its component vectors
def sentvec(s):
    sent = nlp(s)
    return meanv([w.vector for w in sent])

sentences = list(doc.sents)

# Takes a list of sentences from a spaCy parse and compares them to an input sentence, sorting them by cosine similarity
def spacy_closest_sent(space, input_str, n=10):
    input_vec = sentvec(input_str)
    return sorted(space,
                  key=lambda x: cosine(np.mean([w.vector for w in x], axis=0), input_vec),
                  reverse=True)[:n]

# Here are the sentences in Dracula closest in meaning to "My favorite food is strawberry ice cream."
for sent in spacy_closest_sent(sentences, "My favorite food is strawberry ice cream."):
    print (sent.text)
    print ('---\n\n')