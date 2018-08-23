from nltk.classify import ClassifierI
from statistics import mode
import utils
import pickle


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
        # print(votes)
        choice_votes = votes.count(mode(votes))
        # print(choice_votes)
        conf = choice_votes / len(votes)
        return conf


classifier_f = open("classifier/naivebayes5k.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("classifier/MNB5k.pickle", "rb")
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("classifier/BNB5k.pickle", "rb")
BNB_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("classifier/LogisticRegression5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("classifier/SGDC5k.pickle", "rb")
SGDC_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("classifier/LinearSVC5k.pickle", "rb")
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("classifier/NuSVC5k.pickle", "rb")
NuSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDC_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)


def get_voted_classifier():
    return voted_classifier


def sentiment(words, dictionary):
    features = utils.to_feature_vector(words, dictionary)
    return voted_classifier.classify(features), voted_classifier.confidence(features)