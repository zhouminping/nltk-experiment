# ref: https://medium.com/parrot-prediction/dive-into-wordnet-with-nltk-b313c480e788

from nltk.corpus import wordnet

syns = wordnet.synsets("program")

print(syns)
# print(syns[0])

# synset
print(syns[0].name())


print(syns[0].lemmas())
# just the word
print(syns[0].lemmas()[0].name())

# definition
print(syns[1].definition())

# examples
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        # print(l)
        # print(l.antonyms())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

# w1 = wordnet.synsets("ship")
# print(w1)
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))
print(w2.wup_similarity(w1))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")
print(w1.wup_similarity(w2))
