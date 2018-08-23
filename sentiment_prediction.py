import sentiment_mod as sent
from nltk.corpus import movie_reviews
import utils

sources = utils.load_movie_reviews("short_reviews/positive.txt", "short_reviews/negative.txt")
all_words = utils.get_all_words(sources, ["J"])
dictionary = utils.load_dictionary(all_words, 5000)

reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

count_high_confidence = 0
count_accuracy_with_high_confidence = 0
count_accuracy_with_low_confidence = 0
for review in reviews:
    s, c = sent.sentiment(review[0], dictionary)
    if c > 0.8:
        count_high_confidence += 1
        if s == review[1]:
            count_accuracy_with_high_confidence += 1
    else:
        if s == review[1]:
            count_accuracy_with_low_confidence += 1

print((count_accuracy_with_low_confidence + count_accuracy_with_high_confidence) / len(reviews) * 100)
print(count_accuracy_with_high_confidence / count_high_confidence * 100)

# r1 = "This movie was awesome! The acting was great, plot was wonderful"
# r2 = "This movie was utter junk. I don't see what the point was at all"
# r3 = "hello, you are ok?"
# print(sent.sentiment(word_tokenize(r3)))
# print(sent.sentiment(word_tokenize(r2)))
