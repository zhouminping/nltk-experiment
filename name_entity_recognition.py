import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

'''
NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
'''

name_entity = ['ORGANIZATION', 'PERSON', 'LOCATION', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'FACILITY', 'GPE']

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)


            # namedEntity = nltk.ne_chunk(tagged, binary=True)
            #
            # for subtree in namedEntity.subtrees(filter=lambda t: t.label() == 'NE'):
            #     print(subtree)

            namedEntity = nltk.ne_chunk(tagged)
            # for subtree in namedEntity.subtrees(filter=lambda t: t.label() in name_entity):
            #     print(subtree)
            print(words)
            print(namedEntity)
            # namedEntity.draw()

    except Exception as e:
        print(str(e))

process_content()