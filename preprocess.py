import numpy as np
import pandas as pd
from contractions import contractions_dict
import re
import string
import csv
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

#data clean


def clean_text(text, remove_stopwords=True):
    text = text.lower()
    text = text.split()
    tmp = []

    #restore contractions
    for word in text:
        if word in contractions_dict:
            tmp.append(contractions_dict[word])
        else:
            tmp.append(word)
    text = ' '.join(tmp)

    text = text.lower()

    #remove puncuation from text
    clean_word_list = []
    for word in text:
        clean_alphabet_list = [
            alphabet for alphabet in word if alphabet not in string.punctuation
        ]
        clean_word = ''.join(clean_alphabet_list)
        clean_word_list.append(clean_word)
    text = ''.join(clean_word_list)

    #remove numbers from text
    text = re.sub('[0-9]+', '', text)
    text = ' '.join(text.split())

    #remove stopwords
    text = text.split()
    stops = set(stopwords.words('english'))
    text = [w for w in text if w not in stops]
    text = ' '.join(text)

    # remove hyphens and white spaces
    text = re.sub('–', '', text)
    text = ' '.join(text.split())

    # Removing unnecessary characters from text
    text = re.sub("(\\t)", ' ', str(text)).lower()
    text = re.sub("(\\r)", ' ', str(text)).lower()
    text = re.sub("(\\n)", ' ', str(text)).lower()

    # remove accented chars ('Sómě Áccěntěd těxt' => 'Some Accented text')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode(
        'utf-8', 'ignore'
    )

    text = re.sub("(__+)", ' ', str(text)).lower()
    text = re.sub("(--+)", ' ', str(text)).lower()
    text = re.sub("(~~+)", ' ', str(text)).lower()
    text = re.sub("(\+\++)", ' ', str(text)).lower()
    text = re.sub("(\.\.+)", ' ', str(text)).lower()

    text = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(text)).lower()

    text = re.sub("(mailto:)", ' ', str(text)).lower()
    text = re.sub(r"(\\x9\d)", ' ', str(text)).lower()
    text = re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(text)).lower()
    text = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM',
                  str(text)).lower()

    text = re.sub("(\.\s+)", ' ', str(text)).lower()
    text = re.sub("(\-\s+)", ' ', str(text)).lower()
    text = re.sub("(\:\s+)", ' ', str(text)).lower()
    text = re.sub("(\s+.\s+)", ' ', str(text)).lower()

    try:
        url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(text))
        repl_url = url.group(3)
        text = re.sub(
            r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, str(text))
    except Exception as e:
        pass

    text = re.sub("(\s+)", ' ', str(text)).lower()
    text = re.sub("(\s+.\s+)", ' ', str(text)).lower()

    return text



train_data = pd.read_csv(
    './dataset/train.csv')
test_data = pd.read_csv(
    './dataset/test.csv')
# train_data.head()
train_data = train_data.drop(['id'], axis=1)
train_data = train_data.reset_index(drop=True)
test_data = test_data.drop(['id'], axis=1)
test_data = test_data.reset_index(drop=True)

#data clean
clean_summaries = []
for summary in train_data.highlights:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))

    
clean_texts = []
for text in train_data.article:
    clean_texts.append(clean_text(text))
del train_data
clean_df = pd.DataFrame()
clean_df['text'] = clean_texts[:110000]
clean_df['summary'] = clean_summaries[:110000]
clean_df['summary'].replace('', np.nan, inplace=True)
clean_df.dropna(axis=0, inplace=True)

clean_df['summary'] = clean_df['summary'].apply(lambda x: '<sostok>' + ' ' + x + ' ' + '<eostok>')
del clean_texts
del clean_summaries
clean_df.to_csv('cleaned_data.csv')
