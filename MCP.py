#Version 0
################ Lab 2A 9.4 Section ##################
# Clean Tweets
######################################################
'''
from nltk.corpus import stopwords
import string
import re
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r' )
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile( '[%s]'% re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('' , w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words( 'english' ))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load the document
filename = 'txt_sentoken/neg/compund_neg_27k.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)

print('\n\n')
filename = 'txt_sentoken/pos/compund_pos_27k.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)

'''


################ Lab 2A 9.5 Section ##################
# Develop Vocabulary
######################################################
'''

import string
import re
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r' )
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile( '[%s]'% re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('' , w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words( 'english' ))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)


# save list to file
def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w' )
    file.write(data)
    file.close()


# define vocab
vocab = Counter()
# add all docs to vocab
process_docs( 'txt_sentoken/neg' , vocab)
process_docs( 'txt_sentoken/pos' , vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))


# keep tokens with > 5 occurrence
min_occurrence = 4
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
print(len(tokens))

save_list(tokens, 'Vocab_27K_22Aug.txt' )



################ Lab 2A 9.6 Section ##################
# Save Prepared Data
######################################################
'''

import string
import re
from os import listdir
from nltk.corpus import stopwords

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r' )
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile( '[%s]'% re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('' , w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words( 'english' ))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# save list to file
def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w' )
    file.write(data)
    file.close()


# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


# load all docs in a directory
def process_docs(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines




# load vocabulary
vocab_filename = 'Vocab_27K_22Aug.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# prepare negative reviews
negative_lines = process_docs( 'txt_sentoken/neg' , vocab)
save_list(negative_lines, 'negative_tweets_usedForBagOfWords_22Aug.txt' )
# prepare positive reviews
positive_lines = process_docs( 'txt_sentoken/pos' , vocab)
save_list(positive_lines, 'positive_tweets_usedForBagOfWords_22Aug.txt' )
'''


################ Lab 2B 10.4 Section ##################
# Bag-of-Words Representation
######################################################

'''


import string
import re
from os import listdir
from nltk.corpus import stopwords

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r' )
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile( '[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub( '', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words( 'english' ))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

    # load all docs in a directory
def process_docs(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if filename.startswith( 'u_test' ):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines


# load and clean a dataset
def load_clean_dataset(vocab):
    # load documents
    neg = process_docs( 'txt_sentoken/neg' , vocab)
    pos = process_docs( 'txt_sentoken/pos' , vocab)
    docs = neg + pos
    # prepare labels
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

# load the vocabulary
vocab_filename = 'Vocab_27K_22Aug.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# load all training reviews
docs, labels = load_clean_dataset(vocab)
# summarize what we have
print(len(docs), len(labels))

'''

############################# 10.5

'''
import string
import re
from os import listdir
from numpy import array
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r' )
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile( '[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub( '', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words( 'english' ))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def process_docs(directory, vocab, is_train):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith( 'cv9' ):
            continue
        if not is_train and not filename.startswith( 'cv9' ):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines

# load and clean a dataset
def load_clean_dataset(vocab, is_train):
    # load documents
    neg = process_docs( 'txt_sentoken/neg' , vocab, is_train)
    pos = process_docs( 'txt_sentoken/pos' , vocab, is_train)

    docs = neg + pos
    # prepare labels
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
    return docs, labels

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# define the model
def define_model(n_words):
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation= 'relu' ))
    model.add(Dense(1, activation= 'sigmoid' ))
    # compile network
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    # summarize defined model
    model.summary()
    plot_model(model, to_file= 'model.png' , show_shapes=True)
    return model

# load the vocabulary
vocab_filename = 'Vocab_27K_22Aug.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load all reviews
train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab, False)
# create the tokenizer
tokenizer = create_tokenizer(train_docs)
# encode data
Xtrain = tokenizer.texts_to_matrix(train_docs, mode= 'freq' )
Xtest = tokenizer.texts_to_matrix(test_docs, mode= 'freq' )
# define the model
n_words = Xtest.shape[1]
model = define_model(n_words)
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
# loss, acc = model.evaluate(Xtest, ytest, verbose=0)
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print( 'Test Accuracy: %f' % (acc*100))

'''

######################### 10.6 Comparing Word Scoring Methods
'''

import string
import re
from os import listdir
from numpy import array
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from pandas import DataFrame
from matplotlib import pyplot


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r' )
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile( '[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub( '', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words( 'english' ))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def process_docs(directory, vocab, is_train):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith( 'cv9' ):
            continue
        if not is_train and not filename.startswith( 'cv9' ):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines

# load and clean a dataset
def load_clean_dataset(vocab, is_train):
    # load documents
    neg = process_docs( 'txt_sentoken/neg' , vocab, is_train)
    pos = process_docs( 'txt_sentoken/pos' , vocab, is_train)

    docs = neg + pos
    # prepare labels
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
    return docs, labels

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# # define the model
# def define_model(n_words):
#     # define network
#     model = Sequential()
#     model.add(Dense(50, input_shape=(n_words,), activation= 'relu' ))
#     model.add(Dense(1, activation= 'sigmoid' ))
#     # compile network
#     model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
#     # summarize defined model
#     model.summary()
#     plot_model(model, to_file= 'model.png' , show_shapes=True)
#     return model

# define the model
def define_model(n_words):
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation= 'relu' ))
    model.add(Dense(1, activation= 'sigmoid' ))
    # compile network
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    return model

# evaluate a neural network model
def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
    scores = list()
    n_repeats = 10
    n_words = Xtest.shape[1]
    for i in range(n_repeats):
        # define network
        model = define_model(n_words)
        # fit network
        model.fit(Xtrain, ytrain, epochs=10, verbose=0)
        # evaluate
        _, acc = model.evaluate(Xtest, ytest, verbose=0)
        scores.append(acc)
        print( '%d accuracy: %s' % ((i+1), acc))
    return scores

# prepare bag of words encoding of docs
def prepare_data(train_docs, test_docs, mode):
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)
    # encode training data set
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    # encode training data set
    Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
    return Xtrain, Xtest


# load the vocabulary
vocab_filename = 'Vocab_27K_22Aug.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load all reviews
train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab, False)
# run experiment
modes = [ 'binary' , 'count' , 'tfidf' , 'freq' ]
results = DataFrame()
for mode in modes:
    # prepare data for mode
    Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
    # evaluate model on data for mode
    results[mode] = evaluate_mode(Xtrain, ytrain, Xtest, ytest)
# summarize results

print(results.describe())
# plot results
results.boxplot()
pyplot.show()
'''

######################### 10.7 Predicting TWeets
'''
import string
import re
from os import listdir
from numpy import array
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r' )
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile( '[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub( '', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words( 'english' ))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines

# load and clean a dataset
def load_clean_dataset(vocab):
    # load documents
    neg = process_docs( 'txt_sentoken/neg' , vocab)
    pos = process_docs( 'txt_sentoken/pos' , vocab)

    docs = neg + pos
    # prepare labels
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
    return docs, labels

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# # define the model
# def define_model(n_words):
#     # define network
#     model = Sequential()
#     model.add(Dense(50, input_shape=(n_words,), activation= 'relu' ))
#     model.add(Dense(1, activation= 'sigmoid' ))
#     # compile network
#     model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
#     # summarize defined model
#     model.summary()
#     plot_model(model, to_file= 'model.png' , show_shapes=True)
#     return model



# evaluate a neural network model
def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
    scores = list()
    n_repeats = 10
    n_words = Xtest.shape[1]
    for i in range(n_repeats):
        # define network
        model = define_model(n_words)
        # fit network
        model.fit(Xtrain, ytrain, epochs=10, verbose=0)
        # evaluate
        _, acc = model.evaluate(Xtest, ytest, verbose=0)
        scores.append(acc)
        print( '%d accuracy: %s' % ((i+1), acc))
    return scores

# # define the model
def define_model(n_words):
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation= 'relu' ))
    model.add(Dense(1, activation= 'sigmoid' ))
    # compile network
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    # summarize defined model
    model.summary()
    plot_model(model, to_file= 'Twitter Data Model.png' , show_shapes=True)
    return model

# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, model):
    # clean
    tokens = clean_doc(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode= 'binary' )
    # predict sentiment
    yhat = model.predict(encoded, verbose=0)
    # retrieve predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'


# load the vocabulary
vocab_filename = 'Vocab_27K_22Aug.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load all reviews
train_docs, ytrain = load_clean_dataset(vocab)
test_docs, ytest = load_clean_dataset(vocab)
# create the tokenizer
tokenizer = create_tokenizer(train_docs)
# encode data
Xtrain = tokenizer.texts_to_matrix(train_docs, mode= 'binary' )
Xtest = tokenizer.texts_to_matrix(test_docs, mode= 'binary' )
# define network
n_words = Xtrain.shape[1]
model = define_model(n_words)
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)

# TwitterNegDataFile = open("txt_sentoken/neg/cv9.txt" , "r")

cleanTweets  = open("/Users/aayushshah/Documents/v6/Splitted_txt_sentoken/cv9_compound_neg_27K.txt" , "r")
with open('Test_Tweet_Sentiments_MCP.txt', 'w') as readmefile:

	for line in cleanTweets:
		percent, sentiment = predict_sentiment(line, vocab, tokenizer, model)
		readmefile.write('%sSentiment: %s (%.3f%%)' % (line, sentiment, percent*100) + '\n')

# with open('Negative_Twitter_Sentiment.txt', 'w') as readmefile:

	# for line in TwitterNegDataFile:
	# 	percent, sentiment = predict_sentiment(line, vocab, tokenizer, model)
	# 	readmefile.write('Review: [%s]\nSentiment: %s (%.3f%%)' % (line, sentiment, percent*100) + '\n')

# TwitterPosDataFile = open("txt_sentoken/pos/cv9.txt" , "r")
# with open('Positive_Twitter_Sentiment.txt', 'w') as readmefile:

	# for line in TwitterPosDataFile:
	# 	percent, sentiment = predict_sentiment(line, vocab, tokenizer, model)
	# 	readmefile.write('Review: [%s]\nSentiment: %s (%.3f%%)' % (line, sentiment, percent*100) + '\n')
'''
