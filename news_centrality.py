import re
import requests
import socket
import datetime
import json

import hanja
import emoji
from soynlp.normalizer import repeat_normalize

from kiwipiepy import Kiwi
from mecab import MeCab

import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings(action='ignore')

pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
picture_pattern = re.compile(
    r'사진=[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+')

email_pattern = re.compile(
    r'[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')

def clean(sentence):
    def clean_list(x):
        try:
            x = hanja.translate(x, 'substitution') #한자 -> 한글
            x = re.sub(r'\[[^)]+\]',"", x)
            x = re.sub(r'\([^)]+\)',"", x)
            x = pattern.sub(' ', x)
            x = emoji.replace_emoji(x, replace='') #emoji 삭제
            x = url_pattern.sub('', x)
            x = picture_pattern.sub('', x)
            x = email_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
        except:
            x= ""
        return x
    if isinstance(sentence, list):
        kiwi = Kiwi()
        sentence = list(map(lambda x: x.text, kiwi.split_into_sents(sentence)))
        if sentence[-1].endswith('..'):
            sentence.pop()    
        result = [clean_list(s) for s in sentence]
    else:
        result = clean_list(sentence)
    return result

def load_news():
    df = pd.read_csv('/home/kyeonghun.kim/NSI_project/total_data/total_news.csv', index_col=0)
    df = df[['일자', '제목', 'URL']].reset_index(drop=True)

    total_centrality = pd.read_csv('./NSI_dash/data/total_centrality.csv', index_col=0)
    df = df[df['일자']>sorted(total_centrality['일자'].unique())[-1]]
    df['새제목'] = df['제목'].apply(clean)

    return df, total_centrality

def calculate_centrality(df, total_centrality):
    mecab = MeCab()
    df['pos'] = df['새제목'].apply(lambda x: " ".join(mecab.nouns(x)))
    total = pd.DataFrame([], columns=['일자','제목','eigen','degree','result'])
    for date in df['일자'].unique():
        vect = CountVectorizer()
        tmp_date = df[df['일자']==date]
        vects = vect.fit_transform(tmp_date['pos'])
        if tmp_date.shape[0] <5:
            continue
        
        td = pd.DataFrame(vects.todense())
        td.columns = vect.get_feature_names()
        term_document_matrix = td.T
        term_document_matrix.columns = ['Doc '+str(i) for i in range(1, len(td)+1)]
        term_document_matrix['total_count'] = term_document_matrix.sum(axis=1)

        term_document_matrix = term_document_matrix.sort_values(by ='total_count',ascending=False) 
        term_document_matrix = term_document_matrix.drop(columns=['total_count'])
        term_document_matrix = term_document_matrix.T

        co_occ = term_document_matrix.dot(term_document_matrix.T)
        co_occ2 = (co_occ>0).astype(int)
        source=[]
        target=[]
        weight=[]
        for i in range(len(co_occ2)):
            for j in range(len(co_occ2)):
                if i>=j:
                    continue
                source.append(i)
                target.append(j)
                weight.append(co_occ2.iloc[i,j])
        edges = pd.DataFrame(
            {
                "source": source,
                "target": target,
                "weight": weight,
            }
        )

        G = nx.from_pandas_edgelist(edges, edge_attr=True)

        ec = nx.eigenvector_centrality(G, max_iter=500, weight='weight')
        degree = []
        for i in range(len(co_occ)):
            degree.append(sum(co_occ.iloc[i])-co_occ.iloc[i,i])
        tmp_date['eigen'] = ec.values()
        tmp_date['degree'] = degree

        # Define two arrays
        ec_values = np.array(list(ec.values()))
        degree_values = np.array(degree)

        # Perform elementwise dot product
        result = ec_values* degree_values

        tmp_date['result'] = result
        total = pd.concat([total, tmp_date[['일자','제목','URL','eigen','degree','result']].reset_index(drop=True)],ignore_index=True)

    total_centrality = pd.concat([total_centrality, total],ignore_index=True)
    total_centrality.to_csv('/home/kyeonghun.kim/youtube_script/NSI_dash/data/total_centrality.csv')


if __name__ == '__main__':
    try:
        import os
        os.chdir('/home/kyeonghun.kim/NCI_project')
        df, total_centrality = load_news()
        calculate_centrality(df, total_centrality)
    except:
        msg = "ERROR from news_centrality.py"
        contact_to = '\nContact to @kyeonghun.kim'