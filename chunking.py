import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

'''
+ = match 1 or more
? = match 0 or 1 repetitions.
* = match 0 or MORE repetitions
. = Any character except a new line
'''

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            # print(words)
            tagged = nltk.pos_tag(words)
            # print(tagged)

            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?} """
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            # for subtree in chunked.subtrees():
            #     print(subtree)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)
            # print(chunked)
            # chunked.draw()

    except Exception as e:
        print(str(e))

process_content()