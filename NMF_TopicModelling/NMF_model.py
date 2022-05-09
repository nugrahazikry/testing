# for general purpose
import pandas as pd
import re
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from flask import Flask, jsonify
import pandas as pd

# for pre-processing data
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# Gensim for Topic modelling
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

def custom_stopword():
    stop_factory = StopWordRemoverFactory().get_stop_words()
    more_stopword = pd.read_csv('stopword_tweet_pilkada_DKI_2017.csv')
    more_stopword = more_stopword.kata_stopword.tolist()
    
    # Merge stopword
    added_new_data = list(set(stop_factory + more_stopword))
    
    dictionary_stopword = ArrayDictionary(added_new_data)
    new_factory = StopWordRemover(dictionary_stopword)
    
    return new_factory

def text_cleaning(text, stop_word_factory):
    # remove link from tweets
    text_clean = re.sub(r'\S*http\S*', '', text)    
    # remove hastag
    text_clean = re.sub(r'#\S*', '', text_clean)    
    # remove mention
    text_clean = re.sub(r'@\S*', '', text_clean)
    # remove emoticon
    text_clean = re.sub(r'<[\w|\s|\-|&]*>', '', text_clean)
    # remove punctuation
    text_clean = text_clean.translate(str.maketrans('','', string.punctuation + '“' + '”')).lower()    
    # remove digit from text
    text_clean = re.sub(r"\d+", "", text_clean)    
    # remove tab, new line
    text_clean = re.sub(r"\s+", " ", text_clean)    
    # remove stopword for bahasa
    text_clean = stop_word_factory.remove(text_clean)    
    return text_clean

def gen_words(texts):
    final = []
    drop_index = []
    
    for index,text in enumerate(texts):
        new = simple_preprocess(text, deacc=True, min_len=3, max_len=25)
        if new != []:
            final.append(new)
        else:
            drop_index.append(index)
        
    return final, drop_index

def cleantext():
    df_tw = pd.read_csv('dataset_tweet_sentiment_pilkada_DKI_2017.csv')
    df_tw_clean = df_tw.copy()
    new_stop_word = custom_stopword()

    df_tw_clean['CleanedText'] = df_tw_clean['Text Tweet'].apply(text_cleaning, stop_word_factory= new_stop_word)
    df_tw_clean['CleanedText'] = df_tw_clean['CleanedText'].str.strip()
    df_tw_clean = df_tw_clean[df_tw_clean['CleanedText'] != ''].reset_index(drop=True).copy()

    data_words, index_drop = gen_words(df_tw_clean['CleanedText'].values)
    df_tw_clean.drop(index=index_drop, inplace=True)
    df_tw_clean.drop('Id', axis =1, inplace=True)
    df_tw_clean = df_tw_clean.reset_index(drop=True)
    return df_tw_clean

def top_words(topic, n_top_words):
    return topic.argsort()[:-n_top_words - 1:-1]  

def topic_table(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        t = (topic_idx)
        topics[t] = ','.join([feature_names[i] for i in top_words(topic, n_top_words)])

    return pd.DataFrame(topics, index=['Keyword'])

def nmf_model():
  texts = cleantext().CleanedText

  tfidf_vectorizer = TfidfVectorizer(
                      min_df=0.0011,
                      max_df=0.9,
                      max_features=2200,
                      ngram_range=(1, 2), token_pattern= '[a-zA-Z]{3,}')

  tfidf = tfidf_vectorizer.fit_transform(texts)
  tfidf_fn = tfidf_vectorizer.get_feature_names_out()

  nmf = NMF(
      n_components=6,
      init='nndsvd',
      max_iter=7000,
      l1_ratio=0.3,
      solver='cd',
      alpha=0.001,
      tol=1e-5,
      random_state=145)
  
  nmf.fit(tfidf)

  docweights = nmf.transform(tfidf_vectorizer.transform(texts))
  n_top_words = 15
  topic_df = topic_table(nmf, tfidf_fn, n_top_words).T

  df_tw_clean = cleantext()
  df_tw_clean['Dominant_Topic'] = docweights.argmax(axis=1)
  df_tw_clean['Keyword'] = df_tw_clean.Dominant_Topic.map(topic_df.Keyword.to_dict())

  return df_tw_clean

def top_count_word(topic, df_tw_clean, df_countvect, n=15):
  dict_word = df_countvect[df_tw_clean.Dominant_Topic == topic].sum(axis=0).sort_values(ascending=False).nlargest(n).to_dict()
    
  return dict_word

def vectorizer():
  texts = cleantext().CleanedText
  count_vectorizer = CountVectorizer(
                    min_df=0.0011,
                    max_df=0.9,
                    max_features=2200,
                    ngram_range=(1, 2),
                    token_pattern='[a-zA-Z]{3,}')

  count_wm = count_vectorizer.fit_transform(texts)
  count_tokens = count_vectorizer.get_feature_names_out()

  df_tw_clean = nmf_model()

  df_countvect = pd.DataFrame(data = count_wm.toarray(), index = cleantext().index, columns = count_tokens)
  dict_of_word = df_countvect.sum(axis=0).sort_values().to_dict()

  df_bar_plot = pd.DataFrame(columns=['Word', 'Topic', 'Count_Word', 'Total_Word'])

  for i in zip(df_tw_clean.Dominant_Topic.value_counts().index):
    dataplot = top_count_word(i, df_tw_clean, df_countvect, 15)
    df_bar_plot = df_bar_plot.append(
                      {'Word' : list(dataplot.keys()), 
                        'Topic': i, 
                        'Count_Word' : list(dataplot.values()), 
                        'Total_Word' : [dict_of_word[i] for i in dataplot.keys()]}, 
                        ignore_index=True)
  return df_bar_plot

app = Flask(__name__)

@app.route('/')
def barplot():
    NMF_barplot = vectorizer()
    NMF_barplot_list = NMF_barplot.values.tolist()
    json_NMF_barplot = jsonify(NMF_barplot_list)
    return json_NMF_barplot

if __name__ == "__main__":
    app.run(debug=True)