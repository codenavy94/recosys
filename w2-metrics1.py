# Created on Sep 2021
# @author: 임일

# MAE/MAD 계산
def mad(y_true, y_pred):
    import numpy as np
    return np.mean(abs(np.array(y_true) - np.array(y_pred)))
    
predictions = [1,0,3,4]
targets = [1,2,2,4]
mad = mad(targets, predictions)
print(mad)

# sklearn으로 MSE 계산
from typing import TYPE_CHECKING
from sklearn.metrics import mean_squared_error
import numpy as np
predictions = [1,0,3,4]
targets = [1,2,2,4]
mse = mean_squared_error(targets, predictions)
print(f'MSE: {mse}')
# RMSE 계산
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

# 직접 RMSE 계산
def RMSE(y_true, y_pred):
    import numpy as np
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

predictions = [1,0,3,4]
targets = [1,2,2,4]
rmse = RMSE(targets, predictions)
print(rmse)

targets = [1, 100, 40, 20, 44, 98] # 상품ID = TP + FN
predictions = [2, 100, 30, 21] # 구매했을 거라고 예측된 상품 ID = TP + FP

# Target이 binary인 경우: Precision, Recall, F1 계산
def b_metrics(y_true, y_pred):

    import numpy as np

    n_match = len(set(y_true).intersection(set(y_pred)))
    precision = n_match / len(y_pred)
    recall = n_match / len(y_true)
    F1 = 2 * precision * recall / (precision + recall)

    return precision, recall, F1

b_metrics(targets, predictions)
