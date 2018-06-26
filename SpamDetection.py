# SMS SPAM DETECTION USING BAG OF WORDS MODEL
# BY - Omkar Sabnis - 26/06/2018

# IMPORTING ALL THE MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from functools import reduce
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('Data/SMSSpamCollection.txt',sep='\t',header=None)
data.columns = ['label','message']
print('Entries in the dataset:')
print(data.head())

# NUMBER OF HAM AND SPAM MESSAGES
print(data['label'].value_counts())

# PREPROCESSING
def preprocess(text):
    output = re.findall('[A-Za-z]+', text.lower())
    return output
data['preprocessed'] = data.message.apply(lambda text:' '.join(preprocess(text)))
print('Processed Dataset:')
print(data.head())

# VISUALIZATIONS
spam_words = reduce(lambda x,y : x+" "+y, data[data.label == 'spam'].preprocessed)
ham_words = reduce(lambda x,y : x+" "+y, data[data.label == 'ham'].preprocessed)
spam_freq = Counter(spam_words.split())
ham_freq = Counter(ham_words.split())
sfdf = pd.DataFrame(spam_freq.most_common(), columns = ['word', 'frequency'])
print('Word Frequencies:')
print(sfdf.head())

# PLOTTING THE WORDS THAT OCCUR MOSTLY IN SPAM MESSAGES
fig, ax = plt.subplots(figsize = (30, 15))
sfdf[:20].plot(x = 'word', y = 'frequency', kind = 'bar', width = 0.8, ax = ax, fontsize = 25)
ax.set_xlabel('Words')
ax.set_ylabel('Frequency')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x(),p.get_height()+1.0), fontsize = 25)
#plt.show()

#PLOTTING THE WORDS THAT OCCUR MOSTLY IN REAL MESSAGES
hfdf = pd.DataFrame(ham_freq.most_common(), columns = ['word', 'frequency'])
print('Word Frequencies:')
print(hfdf.head())
fig, ax = plt.subplots(figsize = (30, 15))
hfdf[:20].plot(x = 'word', y = 'frequency', kind = 'bar', ax = ax, fontsize = 25, color = 'b')
ax.set_xlabel('Words')
ax.set_ylabel('Frequency')

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()-0.1, p.get_height()+1.0), fontsize = 25)

#plt.show()

# VISUALIZING MESSAGE LENGTHS
data['length'] = data.preprocessed.apply(len)
print('Message Lengths:')
print(data.head())
df1 = data[data['label'] == 'ham'].length
data.hist(column = 'length', by ='label', bins = 50, figsize = (11, 5))
#plt.show()

# BAG OF WORDS MODEL
x_train, x_test, y_train, y_test = train_test_split(data.preprocessed, data.label, test_size = 0.1, random_state = 2018)
print(x_train.shape, y_test.shape)
clf = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB())])

clf.fit(x_train, y_train)
cvs = cross_val_score(clf, x_train, y_train, cv = 10, verbose = 0, n_jobs = 4)

print("Accuracy : {} +-{}".format(round(cvs.mean(), 2), round(cvs.std(), 2)))
