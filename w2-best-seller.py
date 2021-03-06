# Created on Sep 2021
# author: 임일
# modified by codenavy94

# Load the u.user file into a dataframe
import pandas as pd
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/RecoSys/Data/u.user', sep='|', names=u_cols, encoding='latin-1')
users = users.set_index('user_id')
users.head()

# Load the u.data file into a dataframe
import pandas as pd
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', sep='\t', names=r_cols, encoding='latin-1') 
ratings = ratings.set_index('user_id')
ratings.head()

# Load the u.item file into a dataframe
import pandas as pd
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies.set_index('movie_id')
movies.head()

# Best-seller recommender
import numpy as np

def RMSE(y_true, y_pred):
    import numpy as np
    return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))

# 영화별 평점평균 구하기, len(movie_mean) = 1682
movie_mean = ratings.groupby(['movie_id'])['rating'].mean()

def recom_movie1(n_items=5):
    movie_sort = movie_mean.sort_values(ascending=False)[:n_items]
    recom_movies = movies.loc[movie_sort.index] # size = (n_items, genre col 23개)
    recommendations = recom_movies['title']
    return recommendations

def recom_movie2(n_items=5):
    return movies.loc[movie_mean.sort_values(ascending=False)[:n_items].index]['title']

print(f'Recommended movies: {recom_movie1(10)}')

# best-seller model의 정확도(rmse) 계산
# y_pred는 특정 영화에 대한 user 943명 전체의 평균
# 각 user가 평가한 영화에 대한 정보만 rating에 있으므로
# 각 user가 평가한 영화 개수만큼만 y_pred를 구해주고, y_true, y_pred의 차이를 구하게 됨.
rmse = []
for user in set(ratings.index): # 모든 유저 943명에 대하여
    y_true = ratings.loc[user][['movie_id', 'rating']] # 한 user가 평가한 모든 movie_id와 rating
    y_pred = movie_mean.loc[ratings.loc[user]['movie_id']] # 위 movie_id들에 대한 예측평점
    accuracy = RMSE(y_true['rating'], y_pred) # 한 명의 user에 대한 rmse
    rmse.append(accuracy) # 최종적으로 rmse의 사이즈는 (943, )
print(np.mean(rmse))
