# binary classification

import nltk
from nltk.tokenize import word_tokenize
import random
import utils
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

# MultinomiaNB: multiple categories
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


reviews = utils.load_movie_reviews("short_reviews/positive.txt", "short_reviews/negative.txt")
all_words = utils.get_all_words(reviews, ["J"])
dictionary = utils.load_dictionary(all_words, 5000)


featuresets = [(utils.to_feature_vector(word_tokenize(review), dictionary), category) for (review, category) in reviews]
random.shuffle(featuresets)
# print(len(featuresets))
# print(featuresets)


training_set = featuresets[:10000]
testing_set = featuresets[10000:]


# naive bayes classifier
# posterior = prior occurrences * likelihood / evidence
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

save_classifier = open("classifier/naivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("classifier/MNB5k.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier accuracy percent: ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("classifier/BNB5k.pickle", "wb")
pickle.dump(BNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("classifier/LogisticRegression5k.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDC_classifier accuracy percent: ", (nltk.classify.accuracy(SGDC_classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("classifier/SGDC5k.pickle", "wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier accuracy percent: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
# classifier.show_most_informative_features(15)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("classifier/LinearSVC5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("classifier/NuSVC5k.pickle", "wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDC_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print("voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[2][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[5][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)