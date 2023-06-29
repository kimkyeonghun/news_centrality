import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import os
import re

import socket
import datetime
import json
import requests

import spacy

nlp = spacy.load('en_core_web_sm')


def clean_text(text):
    text = " ".join(text.split('\n'))
    pattern = r'\([^)]*\)'
    text = re.sub(pattern=pattern, repl='', string=text).strip()
    if text.find('- ') != -1:
        text = text[text.find('- ')+2:]
    elif text.find(' -- ') != -1:
        text = text[text.find(' -- ')+4:]
    elif text.find(' – ') != -1:
        text = text[text.find(' – ')+3:]
    return text


def load_news():
    total_centrality = pd.read_csv(
        './NSI_dash/data/en_total_centrality.csv', index_col=0)

    NAS_PATH = './kyeonghun.kim/investing'
    df = pd.DataFrame([])
    for file in os.listdir(NAS_PATH):
        if file.startswith('bullpen') and 'research' in file:
            tmp = pd.read_csv(os.path.join(NAS_PATH, file), index_col=0)
            tmp = tmp.drop_duplicates(['data_id'])
            tmp = tmp.dropna().reset_index(drop=True)
            df = pd.concat([df, tmp])

    gather_ticker = []
    for u_id in df[df.duplicated(['data_id']) == True].data_id.unique():
        gather_ticker.append(
            (u_id, ",".join(df[df.data_id == u_id].ticker.tolist())))

    df = df.drop_duplicates(['data_id'])
    for d_id, tickers in gather_ticker:
        df.loc[df.data_id == d_id, 'ticker'] = tickers

    df = df.reset_index(drop=True)

    df['content'] = df['content'].apply(clean_text)
    df = df.sort_values('release_time').reset_index(drop=True)

    df = df[['news_title', 'release_time']]

    commodities_news = pd.read_csv(
        './news_dataset/en/investing/news_commodities.csv', index_col=0)
    commodities_investing = pd.read_csv(
        './news_dataset/en/investing/invests_commodities.csv', index_col=0)
    concat_commodities = pd.merge(
        commodities_investing, commodities_news, on='data_id')
    concat_commodities = concat_commodities[['news_title', 'release_time']]

    news_df = pd.concat([df, concat_commodities])
    news_df['release_time'] = pd.to_datetime(news_df['release_time'])
    news_df['release_time'] = news_df['release_time'].dt.strftime('%Y-%m-%d')

    reuter = pd.read_csv(
        '/home/kyeonghun.kim/GICS_concept/pred/reuter_pred_results.csv', index_col=0)
    reuter = reuter[['title', 'time']]
    reuter.rename(columns={'title': 'news_title',
                  'time': 'release_time'}, inplace=True)

    nytimes = pd.read_csv(
        '/home/kyeonghun.kim/GICS_concept/pred/nytimes_pred_results.csv', index_col=0)
    nytimes = nytimes[['new_title', 'published_date']]
    nytimes.rename(columns={'new_title': 'news_title',
                   'published_date': 'release_time'}, inplace=True)

    total_df = pd.concat([news_df, reuter, nytimes], ignore_index=True)

    total_df = total_df.sort_values('release_time').reset_index(drop=True)
    total_df['release_time'] = total_df['release_time'].apply(
        lambda x: "".join(x.split("-")))
    total_df = total_df.drop_duplicates(subset=['news_title']).reset_index(drop=True)
    return total_df, total_centrality


def calculate_centrality(df, total_centrality):
    parsed_df = df[df.release_time > str(sorted(total_centrality['일자'].unique())[-1])]
    parsed_df['pos'] = parsed_df['news_title'].apply(
        lambda x: " ".join(map(lambda y: y.text, nlp(x).noun_chunks)))
    total = pd.DataFrame(
        [], columns=['release_time', 'news_title', 'eigen', 'degree', 'result'])
    for date in parsed_df['release_time'].unique():
        vect = CountVectorizer(stop_words='english')
        tmp_date = parsed_df[parsed_df['release_time'] == date]
        try:
            vects = vect.fit_transform(tmp_date['pos'])
        except:
            continue
        if tmp_date.shape[0] < 5:
            continue

        td = pd.DataFrame(vects.todense())
        td.columns = vect.get_feature_names()
        term_document_matrix = td.T
        term_document_matrix.columns = [
            'Doc '+str(i) for i in range(1, len(td)+1)]
        term_document_matrix['total_count'] = term_document_matrix.sum(axis=1)

        term_document_matrix = term_document_matrix.sort_values(
            by='total_count', ascending=False)
        term_document_matrix = term_document_matrix.drop(
            columns=['total_count'])
        term_document_matrix = term_document_matrix.T

        co_occ = term_document_matrix.dot(term_document_matrix.T)
        co_occ2 = (co_occ > 0).astype(int)
        source = []
        target = []
        weight = []
        for i in range(len(co_occ2)):
            for j in range(len(co_occ2)):
                if i >= j:
                    continue
                source.append(i)
                target.append(j)
                weight.append(co_occ2.iloc[i, j])
        edges = pd.DataFrame(
            {
                "source": source,
                "target": target,
                "weight": weight,
            }
        )

        G = nx.from_pandas_edgelist(edges, edge_attr=True)

        ec = nx.eigenvector_centrality(G, max_iter=5000, weight='weight')
        degree = []
        for i in range(len(co_occ)):
            degree.append(sum(co_occ.iloc[i])-co_occ.iloc[i, i])
        tmp_date['eigen'] = ec.values()
        tmp_date['degree'] = degree

        # Define two arrays
        ec_values = np.array(list(ec.values()))
        degree_values = np.array(degree)

        # Perform elementwise dot product
        result = ec_values * degree_values

        tmp_date['result'] = result
        total = pd.concat([total, tmp_date[['release_time', 'news_title', 'eigen',
                          'degree', 'result']].reset_index(drop=True)], ignore_index=True)

    total.rename(columns={"release_time": '일자',
                 'news_title': '제목'}, inplace=True)
    total['일자'] = total['일자'].apply(lambda x: "".join(x.split("-")))
    total_centrality = pd.concat([total_centrality, total], ignore_index=True)



if __name__ == '__main__':
    try:
        df, total_centrality = load_news()
        calculate_centrality(df, total_centrality)
    except:
        msg = "ERROR from news_centrality_en.py"
        contact_to = '\nContact to @kyeonghun.kim'
