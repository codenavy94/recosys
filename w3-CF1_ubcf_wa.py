# Created on Sep 2021
# author: 임일
# modified by codenavy94

# cf_simple Model Score: 1.0165507847359563

import numpy as np
import pandas as pd

# Read rating data
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item_num', sep='|', names=i_cols, encoding='latin-1')
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
# 1) 사용자간의 유사도(weighted average에서의 가중치) 계산
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix.copy().fillna(0) # null값 0으로 채우기
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy) # numpy array (943, 943)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# Pearson correlation coefficient 사용
# matrix_dummy = rating_matrix.copy().fillna(0).T # .corr()은 column끼리의 유사도 계산
# user_similarity = matrix_dummy.corr(method='pearson')
# user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# RMSE 계산을 위한 함수
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def score(model):
    id_pairs = zip(test['user_id'], test['movie_id'])
    y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])
    y_true = np.array(test['rating'])
    return RMSE(y_true, y_pred)

# 모든 영화의 (movie_id) 가중평균 rating을 계산하는 함수, 
# 가중치는 주어진 사용자와 다른 사용자 간의 유사도(user_similarity)
def cf_simple(user_id, movie_id):
    if movie_id in rating_matrix:   # 해당 movie_id가 rating_matrix에 존재하는지 확인
        # "현재" 사용자와 다른 사용자 간의 similarity 가져오기
        sim_scores = user_similarity[user_id] # user_id와 다른 943명 사용자들 각각의 유사도: (943,)
        # "현재" 영화에 대한 모든 사용자의 rating값(null값 포함) 가져오기
        movie_ratings = rating_matrix[movie_id] # dataframe[] -> column 기준 인덱싱
        # "현재" 영화를 평가하지 않은 사용자의 index 가져오기
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        # "현재" 영화를 평가하지 않은 사용자의 rating (null) 제거
        movie_ratings = movie_ratings.dropna()
        # "현재" 영화를 평가하지 않은 사용자의 similarity값 제거
        sim_scores = sim_scores.drop(none_rating_idx)
        # "현재" 영화를 평가한 모든 사용자의 가중평균값 구하기
        # (sim_scores, movie_ratings의 내적) / (sim_scores=weight의 합)
        mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
    else:  #해당 movie_id가 없으므로 기본값 3.0을 예측치로 돌려 줌
        mean_rating = 3.0
    return mean_rating

# 정확도 계산
print(f"cf_simple Model Score: {score(cf_simple)}")


###################### 추천하기 ######################
# 추천을 위한 데이터 읽기 (추천을 위해서는 전체 데이터를 읽어야 함: 정확도 계산이 목적이 아니므로)
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings.drop('timestamp', axis=1)
rating_matrix = ratings.pivot(values='rating', index='user_id', columns='movie_id')

# 영화 제목 가져오기
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item_num', sep='|', names=i_cols, encoding='latin-1')
movies = movies[['movie_id', 'title']]
movies = movies.set_index('movie_id')

# Cosine similarity 계산
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# 추천하기
def recommender(user, n_items=10):
    # "현재" 사용자의 모든 아이템에 대한 예상 평점 계산
    predictions = []
    rated_index = rating_matrix.loc[user][rating_matrix.loc[user].notnull()].index  # 이미 평가한 영화 확인, rating_matrix.loc[user].notnull()은 True, False로 이루어진 np.array return
    items = rating_matrix.loc[user].drop(rated_index)
    for item_num in items.index:
        predictions.append(cf_simple(user, item_num))                                   # 예상평점 계산
    recommendations = pd.Series(data=predictions, index=items.index, dtype=float)
    recommendations = recommendations.sort_values(ascending=False)[:n_items]        # 예상평점이 가장 높은 영화 선택
    recommended_items = movies.loc[recommendations.index]['title']
    return recommended_items

# 영화 추천 함수 부르기
print(f"2번 user의 10번 영화에 대한 추천 목록: {recommender(2, 10)}")