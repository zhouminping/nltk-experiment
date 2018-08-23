import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

# check the dir of package
# print(nltk.__file__)

# under the dir nltk_data/gutenberg/bible.kjv.txt
sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
print(tok[5:15])