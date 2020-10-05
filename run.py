import re
from collections import Counter, defaultdict
from functools import reduce

ngram_len = 4

def get_data(train,labels):
    X = []
    y = []
    for text,label in zip(train,labels):
        text = clean_data(text.strip())
        label = label.strip()
        X.append(text)
        y.append(label)
    return X,y


def clean_data(doc):
    clean_data = re.sub(r"\d+", " ", doc)
    return clean_data.lower()


def get_ngrams(text,n=int()):
    gram = [text[i:i+n] for i in range(len(text)-1)]
    return gram


def calculate_class_proba(classes):
    class_freq = dict()
    for label in classes:
        if not label in class_freq:
            class_freq[label]=1
        else:
            class_freq[label]+=1
    proba = {k: v / len(classes) for k, v in class_freq.items()}
    return proba


def calculate_conditional(train, classes):
    frequency_table = defaultdict(list)
    ngram_total_count = dict()
    ngram_overall_proba = dict()
    feature_proba = defaultdict(dict)
    # populate a frequency table
    for datapoint,label in zip(train,classes):
        datapoint = get_ngrams(datapoint,ngram_len)
        for ngram in datapoint:
            frequency_table[label].append(ngram)
            # calculate ngram counts in the dataset
            if not ngram in ngram_total_count:
                ngram_total_count[ngram]=1
            else:
                ngram_total_count[ngram]+=1
    # calculate overall ngram proba in the dataset
    for k,v in ngram_total_count.items():
        ngram_overall_proba[k] = v/len(ngram_total_count)
    # calculate ngram proba given the language
    for k,v in frequency_table.items():
        for key,val in Counter(v).items():
            feature_proba[k][key]=val/len(v)
    return ngram_overall_proba,feature_proba


def calculate_likelihood_table(prior, ngram_overall_proba, feature_proba):
    likelihood = defaultdict(dict)
    for k,v in feature_proba.items():
        for key,val in v.items():
            likelihood[k][key] = (val*prior[k])/ngram_overall_proba[key]
    return likelihood


train,labels = get_data(open("x_train.txt"),open("y_train.txt"))
class_proba = calculate_class_proba(labels)
ngram_proba, feature_proba = calculate_conditional(train[0:20000],labels[0:20000])
likelihood = calculate_likelihood_table(class_proba,ngram_proba,feature_proba)

test = get_ngrams("this is very important task I cannot speak right now please call me later",ngram_len)
print(test)
values = defaultdict(list)
for i in test:
    for language,probas in likelihood.items():
        try:
            values[language].append(probas[i])
        except KeyError:
            values[language].append(0.0001)

picker = []
for k,v in values.items():
    print(k,reduce(lambda x, y: x * y, v, 1)*class_proba[k])
    picker.append(reduce(lambda x, y: x * y, v, 1)*class_proba[k])
print(max(picker))