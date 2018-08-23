import pickle
import os.path
import nltk
from nltk.tokenize import word_tokenize


def load_movie_reviews(raw_pos_path, raw_neg_path, review_with_label_path=None):
    if review_with_label_path and os.path.exists(review_with_label_path):
        with open(review_with_label_path, "rb") as review_file:
            reviews = pickle.load(review_file)
        return reviews
    else:
        reviews = []
        with open(raw_pos_path, "r") as pos:
            for review in pos.readlines():
                reviews.append((review, "pos"))
        with open(raw_neg_path, "r") as neg:
            for review in neg.readlines():
                reviews.append((review, "neg"))
        if not review_with_label_path:
            review_with_label_path = "data/reviews.pickle"
        with open(review_with_label_path, "wb") as review_file:
            pickle.dump(reviews, review_file)
        return reviews


def get_all_words(reviews, types):
    all_words = []
    for review in reviews:
        words = word_tokenize(review[0])
        pos = nltk.pos_tag(words)
        for p in pos:
            if p[1][0] in types:
                all_words.append(p[0].lower())
    return all_words


def load_dictionary(words, size, dict_path=None):
    if dict_path and os.path.exists(dict_path):
        with open(dict_path, "rb") as dict_file:
            dictionary = pickle.load(dict_file)
        return dictionary
    else:
        words = nltk.FreqDist(words)
        dictionary = list(words.keys())[:size]
        if not dict_path:
            dict_path = "data/dictionary.pickle"
        with open(dict_path, "wb") as dict_file:
            pickle.dump(dictionary, dict_file)
        return dictionary


def to_feature_vector(words, dictionary):
    features = {}
    for w in dictionary:
        features[w] = (w in words)
    return features
