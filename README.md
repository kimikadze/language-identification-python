# Language Identification with Naive Bayes and N-Gram Features

The code implements a language identification solution based on Naive Bayes algorithm and character N-gram features. The method relies on data prepared by [Thoma](https://arxiv.org/pdf/1801.07779.pdf) for building an N-gram language model. The probabilities are calculated using 117,500 sentences from 235 languages.

Run the program: `python run.py`

Wait until the input prompt. It may take up to a couple of minutes, depending on your machine. Enter the text you want to identify. 

> If the input is too short, the program may fail to recognise the language and will ask for a longer input.

Recommended Python version is 3.8.

Folder structure:

```
run.py //program
labels.csv //language code labels
x_train.txt //training data
y_train.txt //labels for training data
```

> All files should be in the same directory when running the program.

You can change the N-gram length parameter manually in the code, by editing `ngram_len` variable.
