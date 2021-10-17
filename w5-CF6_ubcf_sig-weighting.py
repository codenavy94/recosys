# Created on Sep 2021
# @author: 임일
# Significance weighting
# modified by codenavy94

"""
IDEA:
weighted average를 계산하기 위해 기존에 movie_ratings, sim_scores, others_mean에서
현 movie_id를 평가하지 않은 user_idx를 추출하여 제거하고 평점을 예측했다면,
significance weighting에서는 SIG_LEVEL을 정해두고
1) (rating_binary X rating_binary.T)[user_id]에서 SIG_LEVEL보다 작은 user_idx 또한 제거
2) 기존의 방법대로 weighted average + bias-from-mean을 구해
3) prediction = predicted 편차 + user_mean 으로 최종 예측평점을 구한다.
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

def ubcf_sig_weighting_original(user_id, movie_id, neighbor_size=20):
    import numpy as np
    # 현 user의 평균 가져오기
    user_mean = rating_mean[user_id]
    if movie_id in rating_matrix:
        # 현 user와 다른 사용자 간의 유사도 가져오기
        sim_scores = user_similarity[user_id]
        # 현 movie의 rating 가져오기. 즉, rating_matrix의 열을 추출
        movie_ratings = rating_matrix[movie_id]
        # 모든 사용자의 rating 평균 가져오기
        others_mean = rating_mean
        # 현 user와 다른 사용자 간의 공통 rating개수 가져오기
        common_counts = sig_counts[user_id] # (943, )
        # 현 movie에 대한 rating이 없는 user 선택
        no_rating_TF = movie_ratings.isnull() # True, False로 구성된 np.array (943, )
        # 공통으로 평가한 영화의 수가 SIG_LEVEL보다 낮은 사람 선택
        low_significance_TF = common_counts < SIG_LEVEL # True, False로 구성된 np.array (943, )
        # 평가를 안 하였거나, SIG_LEVEL이 기준 이하인 user 제거
        none_rating_idx = movie_ratings[no_rating_TF | low_significance_TF].index
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
    return prediction


def ubcf_sig_weighting(user_id, movie_id, neighbor_size=20):
    import numpy as np
    # 현 user의 평균 가져오기
    user_mean = rating_mean[user_id]
    if movie_id in rating_matrix:
        # 현 user와 다른 사용자 간의 유사도 가져오기
        sim_scores = user_similarity[user_id]
        # 현 movie의 rating 가져오기. 즉, rating_matrix의 열을 추출
        movie_ratings = rating_matrix[movie_id]
        # 모든 사용자의 rating 평균 가져오기
        others_mean = rating_mean
        # 현 user와 다른 사용자 간의 공통 rating개수 가져오기
        common_counts = sig_counts[user_id] # (943, )
        # 현 movie에 대한 rating이 없는 user 선택
        no_rating_TF = movie_ratings.isnull() # True, False로 구성된 np.array (943, )
        # 공통으로 평가한 영화의 수가 SIG_LEVEL보다 낮은 사람 선택
        low_significance_TF = common_counts < SIG_LEVEL # True, False로 구성된 np.array (943, )
        # 평가를 안 하였거나, SIG_LEVEL이 기준 이하인 user 제거
        none_rating_idx = movie_ratings[no_rating_TF | low_significance_TF].index
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
                # sim_scores를 내림차순으로 정렬하고 k만큼 추출
                sim_scores = sim_scores.sort_values(ascending=False)[:neighbor_size]
                user_idx = sim_scores.index
                # movie_ratings, others_mean도 k에 대해서 추출
                movie_ratings = movie_ratings[user_idx]
                others_mean = others_mean[user_idx]
                # 편차로 예측치 계산
                movie_ratings = movie_ratings - others_mean
                prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
                # 예측값에 현 사용자의 평균 더하기
                prediction = prediction + user_mean
        else:
            prediction = user_mean
    else:
        prediction = user_mean
    return prediction


# 각 사용자 쌍의 공통 rating 수(significance level)를 집계하기 위한 함수
def count_num1():       # for loop 이용
    # 공통 영화 수를 기록할 matrix 만들기
    counts = np.zeros(np.shape(user_similarity))
    # 각 user의 rating 영화를 1로 표시하고 전치
    rating_binary = (rating_matrix > 0).T # 행은 movie_id, 열은 user_id
    # 사용자별 공통 rating 수 세기
    for i, user in enumerate(rating_binary):
        for j, other in enumerate(rating_binary):
            counts[i,j] = np.sum(rating_binary[user] & rating_binary[other])
            # i 사용자와 j 사용자가 모두 1로 표시한 item들의 sum 계산
    return counts

def count_num2():       # matrix 연산 이용
    # 각 user의 rating 영화를 1로 표시
    rating_binary1 = np.array((rating_matrix > 0).astype(float))
    rating_binary2 = rating_binary1.T
    # 사용자별 공통 rating 수 계산
    counts = np.dot(rating_binary1, rating_binary2) # (943, 1631) x (1631, 943) = (943, 943)
    return counts # (943, 943)

sig_counts = count_num2() # (943, 943)
sig_counts = pd.DataFrame(sig_counts, index=rating_matrix.index, columns=rating_matrix.index).fillna(0)

SIG_LEVEL = 4       # minimum significance level 지정
MIN_RATINGS = 2     # 예측치 계산에 사용할 minimum rating 수 지정

print(score(ubcf_sig_weighting_original, 20))
print(score(ubcf_sig_weighting, 20))

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
recommender(3, 20, 35)

'''

