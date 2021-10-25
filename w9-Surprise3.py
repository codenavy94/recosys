# Created on Oct 2021
# Author: 임일
# Surprise 3

# Importing algorithms from Surprise
from surprise import SVD
from surprise import KNNWithZScore
# Importing other modules from Surprise
from surprise import Dataset
from surprise.model_selection import GridSearchCV

# Importing built in MovieLens 100K dataset
data = Dataset.load_builtin('ml-100k')

# KNN 다양한 파라메터 비교
param_grid = {'k': [30, 35, 40, 45, 50],
              'sim_options': {'name': ['pearson_baseline', 'cosine'],
                              'min_support': [1,2],
                              'user_based': [True, False]}
              }
gs = GridSearchCV(KNNWithZScore, param_grid, measures=['rmse'], cv=4)
gs.fit(data)
# 최고 RMSE 출력
print(gs.best_score['rmse'])
# 최고 RMSE의 parameter
print(gs.best_params['rmse'])


# SVD 다양한 파라메터 비교
param_grid = {'n_epochs': [70, 80, 90],
              'lr_all': [0.005, 0.006, 0.007],
              'reg_all': [0.05, 0.07, 0.1]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=4)
gs.fit(data)
# 최고 RMSE 출력
print(gs.best_score['rmse'])
# 최고 RMSE의 parameter
print(gs.best_params['rmse'])
