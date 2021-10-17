# Created on Sep 2021
# @author: 임일
# Time-weighting

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Read rating data
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')

# Rating 데이터를 test, train으로 나누고 train을 full matrix로 변환
x = ratings.copy()
y = ratings['user_id']
train, test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=20)
rating_matrix = train.pivot(values='rating', index='user_id', columns='movie_id')
time_matrix = train.pivot(values='timestamp', index='user_id', columns='movie_id') # 추가됨 (943, 1631)

# RMSE 계산을 위한 함수
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def score(model, neighbor_size=0):
    id_pairs = zip(test['user_id'], test['movie_id'])
    y_pred = np.array([model(user, movie, neighbor_size) for (user, movie) in id_pairs])
    y_true = np.array(test['rating'])
    return RMSE(y_true, y_pred)

# 모든 가능한 사용자 pair의 Cosine similarities 계산
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# 모든 user의 rating 평균 계산 
rating_mean = rating_matrix.mean(axis=1)

def ubcf_sig_weighting(user_id, movie_id, neighbor_size=20):
    import numpy as np
    # 현 user의 평균 가져오기
    user_mean = rating_mean[user_id]
    if movie_id in rating_matrix:
        # 현 user와 다른 사용자 간의 유사도 가져오기
        sim_scores = user_similarity[user_id]
        # 현 user와 다른 사용자 간의 time gap 가져오기
        t_gap = time_gap[user_id]
        # 현 movie의 rating 가져오기. 즉, rating_matrix의 열(크기: 943)을 추출
        movie_ratings = rating_matrix[movie_id]
        # 모든 사용자의 rating 평균 가져오기
        others_mean = rating_mean
        # 현 user와 다른 사용자 간의 공통 rating개수 가져오기
        common_counts = sig_counts[user_id]
        # 현 movie에 대한 rating이 없는 user 선택
        no_rating = movie_ratings.isnull()
        # 공통으로 평가한 영화의 수가 SIG_LEVEL보다 낮은 사람 선택
        low_significance = common_counts < SIG_LEVEL
        # 영화의 평가시점이 너무 먼 사람을 선택
        too_far = t_gap > TIME_GAP
        # 평가를 안 하였거나, SIG_LEVEL, 평가시점이 기준 이하인 user 제거
        none_rating_idx = movie_ratings[no_rating | low_significance | too_far].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        others_mean = others_mean.drop(none_rating_idx)
        if len(movie_ratings) > MIN_RATINGS:    # 충분한 rating이 있는지 확인
            if neighbor_size == 0:              # Neighbor size가 지정되지 않은 경우
                # 편차로 예측치 계산
                movie_ratings = movie_ratings - others_mean
                prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
                # 예측값에 현 사용자의 평균 더하기
                prediction = prediction + user_mean
            else:                               # Neighbor size가 지정된 경우
                # 지정된 neighbor size 값과 해당 영화를 평가한 총사용자 수 중 작은 것으로 결정
                neighbor_size = min(neighbor_size, len(sim_scores))
                # array로 바꾸기 (argsort를 사용하기 위함)
                sim_scores = np.array(sim_scores)
                movie_ratings = np.array(movie_ratings)
                others_mean = np.array(others_mean)
                # 유사도를 순서대로 정렬
                user_idx = np.argsort(sim_scores)
                # 유사도, rating, 평균값을 neighbor size만큼 받기 
                sim_scores = sim_scores[user_idx][-neighbor_size:]
                movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                others_mean = others_mean[user_idx][-neighbor_size:]
                # 편차로 예측치 계산
                movie_ratings = movie_ratings - others_mean
                prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
                # 예측값에 현 사용자의 평균 더하기
                prediction = prediction + user_mean
        else:
            prediction = user_mean
    else:
        prediction = user_mean
    if prediction > 5:
        prediction = 5
    if prediction < 1:
        prediction = 1
    return prediction

# 각 사용자 쌍의 공통 rating 수(significance level)를 집계하기 위한 함수
def count_num():       # matrix 연산 이용
    # 각 user의 rating 영화를 1로 표시
    rating_flag1 = np.array((rating_matrix > 0).astype(float))
    rating_flag2 = rating_flag1.T
    # 사용자별 공통 rating 수 계산
    counts = np.dot(rating_flag1, rating_flag2)
    return counts

def time_gap_calc():
    time_gap = np.zeros(np.shape(user_similarity))
    tg_matrix = time_matrix.T               # 평가 시점 데이터 가져오기
    for i, user in enumerate(tg_matrix):
        for j, other in enumerate(tg_matrix):
            #두 사용자 간에 공통으로 평가한 영화에 대한 time stamp 차이의 평균 계산
            tg_abs = abs((tg_matrix[user] - tg_matrix[other]).dropna())
            time_gap[i,j] = np.mean(tg_abs)
    return time_gap

def time_gap_calc2():
    tg_matrix = np.array(time_matrix)
    return np.nanmean(np.abs(tg_matrix[np.newaxis,:,:] - tg_matrix[:,np.newaxis,:]), axis=2)

sig_counts = count_num()
sig_counts = pd.DataFrame(sig_counts, index=rating_matrix.index, columns=rating_matrix.index).fillna(0)

time_gap = time_gap_calc2()
time_gap = pd.DataFrame(time_gap, index=time_matrix.index, columns=time_matrix.index).fillna(0)

SIG_LEVEL = 4       # minimum significance level 지정. 공통적으로 평가한 영화의 수
MIN_RATINGS = 2     # 예측치 계산에 사용할 minimum rating 수 지정
TIME_GAP = 16000000 # 평가한 시점이 얼마 이상 차이가 날때 제외할지에 대한 기준

score(ubcf_sig_weighting, 35)


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
def recommender(user, n_items=10, neighbor_size=20):
    # 현재 사용자의 모든 아이템에 대한 예상 평점 계산
    predictions = []
    rated_index = rating_matrix.loc[user][rating_matrix.loc[user] > 0].index    # 이미 평가한 영화 확인
    items = rating_matrix.loc[user].drop(rated_index)
    for item in items.index:
        predictions.append(ubcf_sig_weighting(user, item, neighbor_size))       # 예상평점 계산
    recommendations = pd.Series(data=predictions, index=items.index, dtype=float)
    recommendations = recommendations.sort_values(ascending=False)[:n_items]    # 예상평점이 가장 높은 영화 선택
    recommended_items = movies.loc[recommendations.index]['title']
    return recommended_items

# 영화 추천 함수 부르기
recommender(3, 30, 35)

'''


