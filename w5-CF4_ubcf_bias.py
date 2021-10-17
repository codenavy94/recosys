# Created on Sep 2021
# @author: 임일
# Bias-from-mean CF
# modified by codenavy94

"""
IDEA:
weighted average 계산을 할 때 지금까지는
np.dot(sim_scores, movie_ratings) / sim_scores.sum()을 했었는데
기존의 movie_ratings는 현 movie_id에 대한 (평가를 완료한) 사용자들의 rating을 의미했음.
그러나 bias-from-mean을 사용할 때에는
movie_ratings - others_mean 즉, (사용자들의 실제 rating) - (각 사용자들의 평소의 점수(평점평균))
즉, 편차를 사용해서 예측편차를 구하고 해당 예측편차를 현 user의 user_mean에 더해서
최종 예측값을 구함.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Read rating data
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings.drop('timestamp', axis=1)

# Rating 데이터를 test, train으로 나누고 train을 full matrix로 변환
x = ratings.copy()
y = ratings['user_id']
train, test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=12)
rating_matrix = train.pivot(values='rating', index='user_id', columns='movie_id')

# RMSE 계산을 위한 함수
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def score(model):
    id_pairs = zip(test['user_id'], test['movie_id'])
    y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])
    y_true = np.array(test['rating'])
    return RMSE(y_true, y_pred)

# 모든 가능한 사용자 pair의 Cosine similarities 계산
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy) # (943, 943)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# 모든 user의 rating 평균 계산
rating_mean = rating_matrix.mean(axis=1) # (943,)
# column: movie_id(column)들이 mean 하나의 값으로 합쳐진다(왼쪽으로 붙는다)고 생각하면 됨.

# 사용자의 평가경향을 고려한 추천
def ubcf_bias(user_id, movie_id):
    import numpy as np
    # 현 user가 평가한 영화들에 대해 매긴 평점 평균 가져오기 (=현 user는 평균적으로 몇 점을 주나?)
    user_mean = rating_mean[user_id] # (1,)
    if movie_id in rating_matrix:
        # 현 user와 다른 사용자의 유사도 가져오기
        sim_scores = user_similarity[user_id] # (943,)
        # 현 movie에 대한 모든 사용자(943명)들의 rating 가져오기
        movie_ratings = rating_matrix[movie_id] # (943,)
        # 모든 사용자의 rating 평균 가져오기
        others_mean = rating_mean # (943,)
        # 현 movie에 대한 rating이 없는 user 삭제
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        others_mean = others_mean.drop(none_rating_idx) # 현 영화를 평가한 모든 사용자들 각각의 평점평균
        # 편차 예측치 계산
        # 현 movie에 대한 (평가를 한) 사용자들의 평점 - (평가를 한) 사용자들 각각의 평점평균
        movie_ratings = movie_ratings - others_mean
        prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
        # 예측값에 현 사용자의 평균 더하기
        prediction = prediction + user_mean
    else:
        prediction = user_mean # 원래는 3.0을 할당했었는데, 이제는 해당 사용자의 평균으로 default 예측
    return prediction

score(ubcf_bias)


'''

###################### 추천하기 ######################
# 추천을 위한 데이터 읽기 (추천을 위해서는 전체 데이터를 읽어야 함)
import pandas as pd
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
def recommender(user, n_items=10):
    # 현재 사용자의 모든 아이템에 대한 예상 평점 계산
    predictions = []
    rated_index = rating_matrix.loc[user][rating_matrix.loc[user] > 0].index    # 이미 평가한 영화 확인
    items = rating_matrix.loc[user].drop(rated_index)
    for item in items.index:
        predictions.append(ubcf_bias(user, item))                               # 예상평점 계산
    recommendations = pd.Series(data=predictions, index=items.index, dtype=float)
    recommendations = recommendations.sort_values(ascending=False)[:n_items]    # 예상평점이 가장 높은 영화 선택
    recommended_items = movies.loc[recommendations.index]['title']
    return recommended_items

# 영화 추천 함수 부르기
recommender(3, 30)

'''

