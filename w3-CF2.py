# Created on Sep 2021
# author: 임일
# modified by codenavy94

# k-size: 10, model score: 1.027381403614297
# k-size: 20, model score: 1.013277766548689
# k-size: 30, model score: 1.0108864923478802
# k-size: 40, model score: 1.0107979361959278 # best model
# k-size: 50, model score: 1.01154634811628

"""
IDEA:
1) train_test_split으로 데이터를 나눈 뒤
2) user_id를 행, movie_id를 컬럼으로 가지는 rating_matrix(pivot table)를 만든다.
3) 사용자들간의 유사도를 구해 user_similarity(대각행렬)에 저장
4) 특정 영화(movie_id)에 대한 모든 사용자들의 rating -> movie_ratings에 저장
5) 특정 사용자(user_id)와 다른 모든 사용자들의 유사도 -> sim_scores에 저장
6) 특정 영화를 평가하지 않은 사용자들을 drop을 통해 movie_ratings, sim_scores에서 제거
7) sim_scores를 np.argsort로 오름차순 정렬하고, 가장 큰 유사도를 보이는
   k명의 사용자들의 user_idx를 가져온다.
8) 해당 user_idx 묶음으로 k명의 sim_scores, movie_ratings를 가져온다.
9) np.dot(sim_scores, movie_ratings) / sum(sim_scores) 로 가중평균값 계산!
"""

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

# Rating 데이터를 test, train으로 나누고 train을 full matrix로 변환
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']
train, test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=12)
rating_matrix = train.pivot(values='rating', index='user_id', columns='movie_id')

# Train set의 모든 사용자 pair의 Cosine similarities 계산
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy) # row 간의 유사도 비교, (943, 943)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# RMSE 계산을 위한 함수
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def score(model, neighbor_size=0):
    id_pairs = zip(test['user_id'], test['movie_id'])
    y_pred = np.array([model(user, movie, neighbor_size) for (user, movie) in id_pairs])
    y_true = np.array(test['rating'])
    return RMSE(y_true, y_pred)

# Neighbor size를 고려하는 추천
def cf_knn(user_id, movie_id, neighbor_size=20):
    if movie_id in rating_matrix: # movie_id가 train set에 존재하는지 확인
        # 현재 사용자와 다른 사용자 간의 similarity 가져오기
        sim_scores = user_similarity[user_id]
        # 현재 영화에 대한 모든 사용자의 rating값 가져오기
        movie_ratings = rating_matrix[movie_id]
        # 현재 영화를 평가하지 않은 사용자의 index 가져오기
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        # 현재 영화를 평가하지 않은 사용자의 rating (null) 제거
        movie_ratings = movie_ratings.drop(none_rating_idx)
        # 현재 영화를 평가하지 않은 사용자의 similarity값 제거
        sim_scores = sim_scores.drop(none_rating_idx)
        if neighbor_size == 0:               # Neighbor size가 지정되지 않은 경우
            # 현재 영화를 평가한 모든 사용자의 가중평균값 구하기
            mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
        else:                                # Neighbor size가 지정된 경우
            # 지정된 neighbor size 값과 해당 영화를 평가한 총사용자 수 중 작은 것으로 결정
            neighbor_size = min(neighbor_size, len(sim_scores))
            # array로 바꾸기 (argsort를 사용하기 위함)
            sim_scores = np.array(sim_scores)
            movie_ratings = np.array(movie_ratings)
            # 유사도를 (오름차순) 순서대로 정렬하고 그에 해당하는 user번호를 가져오기
            user_idx = np.argsort(sim_scores)
            # 유사도를 neighbor size만큼 뒤에서부터 가져오기(유사도 큰 순서) ex) [-20:]
            sim_scores = sim_scores[user_idx][-neighbor_size:]
            # 영화 rating을 neighbor size만큼 받기
            movie_ratings = movie_ratings[user_idx][-neighbor_size:]
            # 최종 예측값 계산 
            mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
    else:
        mean_rating = 3.0
    return mean_rating

print(f"KNN 30명 model score: {score(cf_knn, 30)}")

for i in (10, 20, 30, 40, 50):
    print(f"k-size: {i}, model score: {score(cf_knn, i)}")

###################### 추천하기 ######################
# 추천을 위한 데이터 읽기 (추천을 위해서는 전체 데이터를 읽어야 함)
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings.drop('timestamp', axis=1)
rating_matrix = ratings.pivot(values='rating', index='user_id', columns='movie_id')

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
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# 추천하기
def recommender(user, n_items=10, neighbor_size=20):
    # 현재 사용자의 모든 아이템에 대한 예상 평점 계산
    predictions = []
    # 이미 해당 사용자에 의해 평점이 매겨진 movie_idx 가져오기
    rated_index = rating_matrix.loc[user][rating_matrix.loc[user] > 0].index    # 이미 평가한 영화 확인
    items = rating_matrix.loc[user].drop(rated_index)
    for item in items.index: # item이란 특정 movie_id
        predictions.append(cf_knn(user, item, neighbor_size)) # items에 있는 각 영화에 대한 평점 예측
    recommendations = pd.Series(data=predictions, index=items.index, dtype=float)
    recommendations = recommendations.sort_values(ascending=False)[:n_items]    # 예상평점이 가장 높은 영화 선택
    recommended_items = movies.loc[recommendations.index]['title']
    return recommended_items

# 영화 추천 함수 부르기
recommender(2, 10, 20)

