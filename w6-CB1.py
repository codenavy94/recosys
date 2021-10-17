# Created on Oct 2021
# author: 임일
# CB기반 CF (Meta data)

"""
IDEA:
1) 영화의 plot에 대한 tf-idf를 통해 영화들간의 cosine_sim을 구한다.
2) 특정 영화 id를 가져오면, sim_scores = cosine_sim[id] 해서 이를 내림차순 정렬,
1:n_of_recomm+1(자기자신 빼기)만큼 영화 idx가져와서
3) movies에서 해당 idx의 title을 검색하여 추천해준다.
"""

import pandas as pd

# Meta data 읽기
movies = pd.read_csv('C:/RecoSys/Data/movies_metadata.csv', encoding='latin-1', low_memory=False)
movies = movies[['id', 'title', 'overview']]
movies = movies.drop_duplicates()
# null 데이터 수정
movies = movies.dropna()
movies['overview'] = movies['overview'].fillna('')

# TfIdfVectorizer 가져오기
from sklearn.feature_extraction.text import TfidfVectorizer

# 불용어를 english로 지정하고 tf-idf 계산
tfidf = TfidfVectorizer(stop_words='english')
#tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies['overview']) # (44300, 74686) 74686은 불용어 제외한 vocab size

# Cosine 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix) # (44300, 44300)
cosine_sim = pd.DataFrame(cosine_sim, index=movies.index, columns=movies.index)

# index-title을 뒤집는다
indices = pd.Series(movies.index, index=movies['title']) # index가 title인 Series -> (44300, )

# 영화제목을 받아서 추천 영화를 돌려주는 함수
def content_recommender(title, n_of_recomm):
    # title에서 영화 index 받아오기
    idx = indices[title]
    # 주어진 영화와 다른 영화의 similarity를 가져온다
    sim_scores = cosine_sim[idx]
    # similarity 기준으로 정렬하고 n_of_recomm만큼 가져오기 (자기자신은 빼기)
    sim_scores = sim_scores.sort_values(ascending=False)[1:n_of_recomm+1]
    # 영화 title 반환
    return movies.loc[sim_scores.index]['title']

# 추천받기
print(content_recommender('The Lion King', 20))
print(content_recommender('The Dark Knight Rises', 20))
