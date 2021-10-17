# Created on Sep 2021
# author: 임일
# IBCF
# modified by codenavy94

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
ratings = ratings.drop('timestamp', axis=1) # size = (100,000, 3)

from sklearn.model_selection import train_test_split

# Rating 데이터를 test, train으로 나누고 train을 full matrix로 변환
x = ratings.copy()
y = ratings['user_id']
train, test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=12)
rating_matrix_t = train.pivot(values='rating', index='movie_id', columns='user_id')

# RMSE 계산을 위한 함수
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def score(model, item_neighbor_size=20):
    id_pairs = zip(test['user_id'], test['movie_id'])
    y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])
    y_true = np.array(test['rating'])
    return RMSE(y_true, y_pred)

# 아이템 pair의 Cosine similarities 계산
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix_t.copy().fillna(0) # movie_id가 행인 matrix
item_similarity = cosine_similarity(matrix_dummy, matrix_dummy) # (1631, 1631)
item_similarity = pd.DataFrame(item_similarity, index=rating_matrix_t.index, columns=rating_matrix_t.index)

def ibcf(user_id, movie_id):
    import numpy as np
    if user_id in rating_matrix_t:          # 사용자가 train set에 있는지 확인
        if movie_id in item_similarity:     # 현재 영화가 train set에 있는지 확인
            # 현재 영화와 다른 영화의 similarity 값 가져오기
            sim_scores = item_similarity[movie_id] # (1631, )
            # 현 사용자의 "모든 영화"에 대한 rating 값 가져오기
            user_rating = rating_matrix_t[user_id] # (1631, )
            # 사용자가 평가하지 않은 영화 index 가져오기
            non_rating_idx = user_rating[user_rating.isnull()].index
            # 사용자가 평가하지 않은 영화 제거
            user_rating = user_rating.dropna()
            # 사용자가 평가하지 않은 영화의 similarity 값 제거
            sim_scores = sim_scores.drop(non_rating_idx)
            # 현 영화에 대한 사용자의 예상 rating 계산, 가중치는 현 영화와 사용자가 평가한 영화의 유사도
            mean_rating = np.dot(sim_scores, user_rating) / sim_scores.sum()
        else:
            mean_rating = 3.0
    else:
        mean_rating = 3.0
    return mean_rating

def ibcf_knn_original(user_id, movie_id, item_neighbor_size=20):
    import numpy as np
    if user_id in rating_matrix_t:          # 사용자가 train set에 있는지 확인
        if movie_id in item_similarity:     # 현재 영화가 train set에 있는지 확인
            # 현재 영화와 다른 영화의 similarity 값 가져오기
            sim_scores = item_similarity[movie_id]
            # 현 사용자의 모든 rating 값 가져오기
            user_rating = rating_matrix_t[user_id]
            # 사용자가 평가하지 않은 영화 index 가져오기
            non_rating_idx = user_rating[user_rating.isnull()].index
            # 사용자가 평가하지 않은 영화 제거
            user_rating = user_rating.dropna()
            # 사용자가 평가하지 않은 영화의 similarity 값 제거
            sim_scores = sim_scores.drop(non_rating_idx)
            if item_neighbor_size == 0:               # Neighbor size가 지정되지 않은 경우
                # 현재 영화를 평가한 모든 사용자의 가중평균값 구하기
                mean_rating = np.dot(sim_scores, user_rating) / sim_scores.sum()
            else:                                # Neighbor size가 지정된 경우
                # 지정된 neighbor size 값과 해당 영화를 평가한 총사용자 수 중 작은 것으로 결정
                item_neighbor_size = min(item_neighbor_size, len(sim_scores))
                # array로 바꾸기 (argsort를 사용하기 위함)
                sim_scores = np.array(sim_scores)
                user_rating = np.array(user_rating)
                # 유사도를 순서대로 정렬
                user_idx = np.argsort(sim_scores)
                # 유사도를 neighbor size만큼 받기
                sim_scores = sim_scores[user_idx][-item_neighbor_size:]
                # 영화 rating을 neighbor size만큼 받기
                user_rating = user_rating[user_idx][-item_neighbor_size:]
                # 최종 예측값 계산 
                mean_rating = np.dot(sim_scores, user_rating) / sim_scores.sum()
        else:
            mean_rating = 3.0
    else:
        mean_rating = 3.0
    return mean_rating

# argsort를 사용하지 않는 방법
def ibcf_knn(user_id, movie_id, item_neighbor_size=20):
    import numpy as np
    if user_id in rating_matrix_t:          # 사용자가 train set에 있는지 확인
        if movie_id in item_similarity:     # 현재 영화가 train set에 있는지 확인
            # 현재 영화와 다른 영화의 similarity 값 가져오기
            sim_scores = item_similarity[movie_id]
            # 현 사용자의 모든 rating 값 가져오기
            user_rating = rating_matrix_t[user_id]
            # 사용자가 평가하지 않은 영화 index 가져오기
            non_rating_idx = user_rating[user_rating.isnull()].index
            # 사용자가 평가하지 않은 영화 제거
            user_rating = user_rating.dropna()
            # 사용자가 평가하지 않은 영화의 similarity 값 제거
            sim_scores = sim_scores.drop(non_rating_idx)
            if item_neighbor_size == 0:               # Neighbor size가 지정되지 않은 경우
                # 사용자가 평가한 영화들 중에서 현재 영화와 다른 모든 영화를 사용하여 계산한 가중평균
                mean_rating = np.dot(sim_scores, user_rating) / sim_scores.sum()
            else:                                # Neighbor size가 지정된 경우
                # 지정된 neighbor size 값과 현재 사용자가 평가한 총 영화 개수 중에서 작은 값으로 결정
                item_neighbor_size = min(item_neighbor_size, len(sim_scores))
                # 유사도를 순서대로 정렬
                sim_scores = sim_scores.sort_values(ascending=False)[:item_neighbor_size]
                k_movie_idx = sim_scores.index
                # 영화 rating을 neighbor size만큼 받기
                user_rating = user_rating[k_movie_idx]
                # 최종 예측값 계산
                mean_rating = np.dot(sim_scores, user_rating) / sim_scores.sum()
        else:
            mean_rating = 3.0
    else:
        mean_rating = 3.0
    return mean_rating

# 정확도 계산
print(score(ibcf_knn_original))
print(score(ibcf_knn))

'''
###################### 추천하기 ######################
# 추천을 위한 데이터 읽기 (추천을 위해서는 전체 데이터를 읽어야 함)
import pandas as pd
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings.drop('timestamp', axis=1)
rating_matrix_t = ratings.pivot(values='rating', index='movie_id', columns='user_id')

# 영화 제목 가져오기
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies[['movie_id', 'title']]
movies = movies.set_index('movie_id')

# Cosine similarity 계산
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix_t.copy().fillna(0)
item_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
item_similarity = pd.DataFrame(item_similarity, index=rating_matrix_t.index, columns=rating_matrix_t.index)

# 추천하기
def recommender(user, n_items=10):
    # 현재 사용자의 모든 아이템에 대한 예상 평점 계산
    predictions = []
    rated_index = rating_matrix_t[user][rating_matrix_t[user] > 0].index     # 이미 평가한 영화 확인
    items = rating_matrix_t[user].drop(rated_index)
    for item in items.index:
        predictions.append(ibcf(user, item))                                 # 예상평점 계산
    recommendations = pd.Series(data=predictions, index=items.index, dtype=float)
    recommendations = recommendations.sort_values(ascending=False)[:n_items] # 예상평점이 가장 높은 영화 선택
    recommended_items = movies.loc[recommendations.index]['title']
    return recommended_items

# 영화 추천 함수 부르기
recommender(3, 30)
'''

