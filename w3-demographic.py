# Created on Sep 2021
# author: 임일
# modified by codenavy94

# Bestseller Model Score: 1.0234701463131335
# cf_gender Model Score: 1.0330308800874282
# cf_occupation Model Score: 1.1131032429674141
# cf_gender_occupation Model Score: 1.1391976012043645

"""
IDEA:
1) train_test_split으로 데이터를 나눈 뒤
2) 성별 or 직업별로 영화id 각각에 대한 평점 평균을 구한다.
3) test set에 있는 user_id의 성별 or 직업 정보를 가져와서 영화 평점평균을 예측해준다.

"""


import numpy as np
import pandas as pd

# Load the u.user file into a dataframe
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/RecoSys/Data/u.user', sep='|', names=u_cols, encoding='latin-1')

# Load the u.items file into a dataframe
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 'unknown', 
          'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 
          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
          'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=i_cols, encoding='latin-1')

# movie ID와 title을 제외한 컬럼 지우기
movies = movies[['movie_id', 'title']]

# Load the u.data file into a dataframe
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', sep='\t', names=r_cols, encoding='latin-1')

# timestamp 지우기
ratings = ratings.drop('timestamp', axis=1)

# Import the train_test_split function
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']
# Train/Test 데이터 나누기 (stratified 방식)
train, test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)

# RMSE계산 함수
def RMSE(y_true, y_pred):
    import numpy as np
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# Baseline model (평균 평점을 예측값을 돌려 줌, 평균이 없을 경우에는 3)
def baseline_model(user_id, movie_id):
    try:
        rating = train_mean[movie_id]
        # test set에서 예측하려는 movie_id가 train set(즉, train_mean)에 없을 수 있음
    except:
        rating = 3.0
    return rating

# 주어진 추천 알고리즘(model)의 RMSE를 계산하는 함수
def score(model):
    # Construct a list of user-movie tuples from the testing dataset
    id_pairs = zip(test['user_id'], test['movie_id']) # test set 25,000개

    # Predict the rating for every user-movie tuple by using the model
    y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])

    # Extract the actual ratings given by the users in the test data
    y_true = np.array(test['rating'])

    # Return the final RMSE score
    return RMSE(y_true, y_pred)

# train data에서 movie_id별 rating의 ("모든" 사용자들의) 평균값
train_mean = train.groupby(['movie_id'])['rating'].mean() # (943, )
print(f"Bestseller Model Score: {score(baseline_model)}")

# Data frame의 pivot함수로 full matrix 구성
rating_matrix = train.pivot(values='rating', index='user_id', columns='movie_id')
rating_matrix.head()

# training set과 사용자 table을 결합(merge), user_id 기준
merged_data = pd.merge(train, users)
merged_data.head()



######Gender######

# movie_id, gender별 평점 평균 계산
# ex) 1번 영화 남자 평균, 1번 영화 여자 평균, ...
gender_mean = merged_data[['movie_id', 'sex', 'rating']].groupby(['movie_id', 'sex'])['rating'].mean()

# user_id를 index로 설정
users = users.set_index('user_id')

# Gender기준 추천 함수
def cf_gender(user_id, movie_id):
    # movie_id가 rating_matrix에 존재하는지 확인
    # why? train_test_split으로 인해 movie_id가 (train으로만 구성한) rating_matrix에 없을 수도 있음
    if movie_id in rating_matrix:
        # 입력된 user_id의 gender 정보 가져옴
        gender = (users.loc[user_id])['sex']
        # 해당 영화에 해당 gender의 평균값이 존재하는지 확인
        if gender in gender_mean[movie_id]:
            # 해당 영화의 해당 gender의 평균값을 예측값으로 함
            gender_rating = gender_mean[movie_id][gender]
        else:
            gender_rating = 3.0
    else: # movie_id가 rating_matrix에 없으면 기본값 3.0을 예측값으로 함
        gender_rating = 3.0
    return gender_rating

print(f"cf_gender Model Score: {score(cf_gender)}")



######Occupation######

occupation_mean = merged_data[['movie_id', 'occupation', 'rating']].groupby(['movie_id', 'occupation'])['rating'].mean()

def cf_occupation(user_id, movie_id):
    if movie_id in rating_matrix:
        occupation = users.loc[user_id]['occupation']
        if occupation in occupation_mean[movie_id]:
            occupation_rating = occupation_mean[movie_id][occupation]
        else:
            occupation_rating = 3.0
    else:
        occupation_rating = 3.0
    return occupation_rating

print(f"cf_occupation Model Score: {score(cf_occupation)}")


######Gender and Occupation######

gen_occ_mean = merged_data[['movie_id', 'sex', 'occupation', 'rating']].groupby(['movie_id', 'sex', 'occupation'])['rating'].mean()

def cf_gender_occupation(user_id, movie_id):
    if movie_id in rating_matrix: # movie_id가 train dataset에 존재하는지 확인
        gender = users.loc[user_id]['sex']
        occupation = users.loc[user_id]['occupation']
        if (gender, occupation) in gen_occ_mean[movie_id]:
            rating_pred = gen_occ_mean[movie_id][(gender, occupation)]
        else:
            rating_pred = 3.0
    else:
        rating_pred = 3.0
    return rating_pred

print(f"cf_gender_occupation Model Score: {score(cf_gender_occupation)}")