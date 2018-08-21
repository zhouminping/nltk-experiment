# import nltk

# nltk.download()

# tokenizing- word tokenizers Vs sentence tokenizers

# lexicon and corporas
# corpora - body of text. ex: medical journals, presidential speeches, english language
# lexicon - words and their means, like dictionary

# investor-speak Vs regular english-speak
# investor-speak 'bull' = someone who is positive about the market
# english-speak 'bull' = scary animal you don't want running at you

from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. " \
               "The sky is pinkish-blue. You should not eat cardboard."

print(sent_tokenize(example_text))
print(word_tokenize(example_text))






