# Created on Sep 2021
# author: 임일
# IBCF binary (precision, recall, F1 구하기)

import numpy as np
import pandas as pd
# Read rating data
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies[['movie_id', 'title']]
movies = movies.set_index('movie_id')
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings.drop('timestamp', axis=1)

from sklearn.model_selection import train_test_split

# Rating 데이터를 test, train으로 나누고 train을 full matrix로 변환
x = ratings.copy()
y = ratings['user_id']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=12)
rating_matrix_t = x_train.pivot(values='rating', index='movie_id', columns='user_id')
x_test = x_test.set_index('user_id')
x_train = x_train.set_index('user_id')

# precision, recall, F1 계산을 위한 함수
def b_metrics(y_true, y_pred):
    n_match = set(y_true).intersection(set(y_pred))
    precision = 0
    recall = 0
    F1 = 0
    if len(y_pred) > 0:          # 분모가 0인지 확인
        precision = len(n_match) / len(y_pred)
    if len(y_true) > 0:          # 분모가 0인지 확인
        recall = len(n_match) / len(y_true)
    if (precision + recall) > 0: # 분모가 0인지 확인
        F1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, F1

def score_binary(model, n_of_recomm=10, ref_size=2):
    precisions = []
    recalls = []
    F1s = []
    for user in set(x_test.index):              # Test set에 있는 모든 사용자 각각에 대해서 실행
        y_true = x_test.loc[user]['movie_id']
        #y_true = x_test.loc[user][x_test.loc[user]['rating'] >= cutline]['movie_id']    # cutline 이상의 rating만 정확한 것으로 간주
        if n_of_recomm == 0:                    # 실제 평가한 영화수 같은 수만큼 추천 
            n_of_recomm = len(y_true)
        y_pred = model(user, n_of_recomm, ref_size)
        precision, recall, F1 = b_metrics(y_true, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
    return np.mean(precisions), np.mean(recalls), np.mean(F1s)

# 아이템 pair의 Cosine similarities 계산
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix_t.copy().fillna(0)
item_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
item_similarity = pd.DataFrame(item_similarity, index=rating_matrix_t.index, columns=rating_matrix_t.index)

def ibcf_binary(user, n_of_recomm=10, ref_size=2):
    rated_index = rating_matrix_t[user][rating_matrix_t[user] > 0].index
    ref_group = rating_matrix_t[user].sort_values(ascending=False)[:ref_size]
    sim_scores = item_similarity[ref_group.index].mean(axis=1)
    sim_scores = sim_scores.drop(rated_index)
    recommendations = sim_scores.sort_values(ascending=False)[:n_of_recomm].index
    return recommendations

# 정확도 계산
score_binary(ibcf_binary, 22, 10)


