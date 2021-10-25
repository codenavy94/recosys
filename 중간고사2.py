import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings.drop('timestamp', axis=1)

x = ratings.copy() # 
y = ratings['user_id']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=12)
rating_matrix = x_train.pivot(values='rating', index='user_id', columns='movie_id') # 500, 1000

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def score(model):
    id_pairs = zip(x_test['user_id'], x_test['movie_id'])                      
    y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])      ### <-- (a)
    y_true = np.array(x_test['rating'])
    return RMSE(y_true, y_pred)

matrix_dummy = rating_matrix.copy().fillna(0)
#user_similarity = np.corrcoef(matrix_dummy)'
#user_similarity = matrix_dummy.T.corr(method = 'pearson') # 500, 500
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)                ### <-- (d)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)
rating_mean = rating_matrix.mean(axis=1)
rating_std = rating_matrix.std(axis=1)

def ubcf_bias(user_id, movie_id):
    user_mean = rating_mean[user_id]
    user_std = rating_std[user_id]
    if movie_id in rating_matrix:
        sim_scores = user_similarity[user_id]                                  ### <-- (b)
        movie_ratings = rating_matrix[movie_id]
        others_mean = rating_mean
        others_std = rating_std
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        others_mean = others_mean.drop(none_rating_idx)
        others_std = others_std.drop(none_rating_idx)
        if others_std > 0:
            movie_ratings = (movie_ratings - others_mean) / others_std
        else:
            movie_ratings = movie_ratings - others_mean
        prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()      ### <-- (c)
        prediction = prediction * user_std + user_mean # 현재 사용자의 평균, 표준편차로 de-normalize
    else:
        prediction = user_mean
    return prediction

score(ubcf_bias)
