from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
# If pos is not supplied, the default is noun, i.e., pos='n'
print(lemmatizer.lemmatize("wrote"))
print(lemmatizer.lemmatize("wrote", pos="v"))
print(lemmatizer.lemmatize("writing", "v"))
print(lemmatizer.lemmatize("writes", "v"))
print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("running", "v"))