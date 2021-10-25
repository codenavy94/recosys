import numpy as np
import pandas as pd
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
rating_matrix = np.array(ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0))
class MF():
    def __init__(self, rating_matrix, K, alpha, beta, iterations, verbose=True):
        self.R = rating_matrix # (500, 1000)
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.verbose = verbose
    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K)) # (500, 20)
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K)) # (1000, 20)
        self.b_u = np.zeros(self.num_users) # 500
        self.b_d = np.zeros(self.num_items) # 1000
        self.b = np.mean(self.R[np.where(self.R != 0)])
        self.samples = [
        (i, j, self.R[i, j])
        for i in range(self.num_users)
            for j in range(self.num_items)
                if self.R[i, j] > 0
        ]
        training_process = []
        for i in range(self.iterations): # 45번
            np.random.shuffle(self.samples)
            self.sgd()
            self.full_matrix = self.full_prediction()
            measure = self.rmse()
            training_process.append((i, measure))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; RMSE = %.4f" % (i+1, measure))
        return training_process
    def rmse(self):
        xs, ys = self.R.nonzero()
        self.predictions = []
        self.errors = []
        for x, y in zip(xs, ys):
            self.predictions.append(self.full_matrix[x, y])
            self.errors.append(self.R[x, y] - self.full_matrix[x, y])
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))
    def sgd(self):
        for i, j, r in self.samples: # self.samples size = 108,000개 원소
            prediction = self.get_prediction(i, j)
            e = (r - prediction)
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])        
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])       ####   <-- (a)
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])        
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    def full_prediction(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_d[np.newaxis,:] + self.P.dot(self.Q.T)

mf = MF(rating_matrix, K=20, alpha=0.001, beta=0.02, iterations=45, verbose=True)
mf.train()                                                                      
