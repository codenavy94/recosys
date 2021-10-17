# Created on Oct 2021
# author: 임일
# Matrix factorization 1 - 원리

import numpy as np
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)            # timestamp 제거
rating_matrix = ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)

# New MF class for training & testing
class MF():
    # Initializing the object
    def __init__(self, rating_matrix, K, alpha, beta, iterations, verbose=True):
        self.R = np.array(rating_matrix)
        # user_id, movie_id를 R의 index와 매칭하기 위한 dictionary 생성
        movie_id_index = []
        index_movie_id = []
        for i, one_id in enumerate(rating_matrix): # for one_id in rating_matrix: -> column(=movie_id) index를 가져오게 됨
            movie_id_index.append([one_id, i])
            index_movie_id.append([i, one_id])
        self.movie_id_index = dict(movie_id_index) # 1682개
        self.index_movie_id = dict(index_movie_id)        
        user_id_index = []
        index_user_id = []
        for i, one_id in enumerate(rating_matrix.T): # rating_matrix.T -> (1682, 943)
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])
        self.user_id_index = dict(user_id_index) # 943개
        self.index_user_id = dict(index_user_id)
        # 다른 변수 초기화
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.verbose = verbose

    def train(self):                             # Training 하면서 test set의 정확도를 계산하는 메소드 
        # Initializing user-feature and movie-feature matrix
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K)) # (943, K)
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K)) # (1682, K)

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()]) # rating이 0이 아닌 것들에 대해서만 평균내기

        # List of training samples
        rows, columns = self.R.nonzero()
        self.samples = [(i,j, self.R[i,j]) for i, j in zip(rows, columns)] # 100,000개

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations): # 1번 반복에 대해서 self.P와 self.Q를 업데이트
            np.random.shuffle(self.samples)
            self.sgd() # iteration 수만큼 돌아감
            rmse = self.rmse()
            training_process.append((i, rmse))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; Train RMSE = %.6f " % (i+1, rmse))
        return training_process

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples: # loop: 100,000번 돌아감
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])

            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])

    # Computing mean squared error
    def rmse(self):
        xs, ys = self.R.nonzero() # R matrix의 0이 아닌 element에 대한 row, column index 정보 가져오기, (총 10만개임)
        self.predictions = []
        self.errors = []
        for x, y in zip(xs, ys):
            prediction = self.get_prediction(x, y)
            self.predictions.append(prediction)
            self.errors.append(self.R[x, y] - prediction)
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))

    # Ratings for user i and moive j
    # Week6 PPT p. 23 첫 번째 수식 참고
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Ratings for user_id and moive_id
    def get_one_prediction(self, user_id, movie_id):
        return self.get_prediction(self.user_id_index[user_id], self.movie_id_index[movie_id])

# Original MF
mf = MF(rating_matrix, K=30, alpha=0.001, beta=0.02, iterations=20, verbose=True)
mf.train()
print(mf.get_one_prediction(1,2))

