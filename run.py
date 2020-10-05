import re
from collections import Counter, defaultdict
from functools import reduce
from operator import itemgetter

ngram_len = 4

def get_data(train,labels):
    """get data structures with cleaned train data and labels """
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
    """generate n-grams of len n for a given text"""
    gram = [text[i:i+n] for i in range(len(text)-1)]
    return gram


def calculate_class_proba(classes):
    """calculate class prior"""
    class_freq = dict()
    for label in classes:
        if not label in class_freq:
            class_freq[label]=1
        else:
            class_freq[label]+=1
    proba = {k: v / len(classes) for k, v in class_freq.items()}
    return proba


def calculate_conditional(train, classes):
    """calculate conditional probabilities for attributes"""
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
    """get likelihood table"""
    likelihood = defaultdict(dict)
    for k,v in feature_proba.items():
        for key,val in v.items():
            likelihood[k][key] = (val*prior[k])/ngram_overall_proba[key]
    return likelihood


def map_language(result, table):
    """map the code of detected language to the language name"""
    language_ids = dict()
    for line in open(table,"r"):
        line = line.strip().split(";")
        language_ids[line[0]] = line[1]
    return language_ids[result]


def detect_language_NB(input_text,likelihood,class_proba):
    """identify the most probable language for a given text using NB"""
    values = defaultdict(list)
    pre_selector = []
    input_text = get_ngrams(clean_data(input_text),ngram_len)
    for i in input_text:
        for language,probas in likelihood.items():
            try:
                values[language].append(probas[i])
            except KeyError:
                values[language].append(0.0001)
    for k,v in values.items():
        pre_selector.append((k,reduce(lambda x, y: x * y, v, 1) * class_proba[k]))
    if max(pre_selector,key=itemgetter(1))[0] == min(pre_selector,key=itemgetter(1))[0]:
        return False
    else:
        return map_language(max(pre_selector,key=itemgetter(1))[0],'labels.csv')


def main():
    train, labels = get_data(open("x_train.txt"), open("y_train.txt"))
    class_proba = calculate_class_proba(labels)
    ngram_proba, feature_proba = calculate_conditional(train, labels)
    likelihood = calculate_likelihood_table(class_proba, ngram_proba, feature_proba)
    print("Calculating probabilities. Hang tight!")
    while True:
        test = input("Enter your text (press Enter to exit): ")
        if test == '':
            print("Buy!")
            exit()
        try:
            if detect_language_NB(test,likelihood,class_proba) != False:
                print("The language of your document is ", detect_language_NB(test,likelihood,class_proba))
            else:
                print("Language cannot be reliably identified. Please try another time.")
        except ValueError:
            print("Please enter some text...")


main()