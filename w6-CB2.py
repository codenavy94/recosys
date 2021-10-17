# Created on Sep 2021
# author: 임일
# CB기반 CF (Plot & crew)

import pandas as pd
import numpy as np

# Meta data 읽기
movies = pd.read_csv('C:/RecoSys/Data/movies_metadata.csv', encoding='latin-1', low_memory=False)
movies = movies[['id', 'title']]
movies = movies.dropna()
credits = pd.read_csv('C:/RecoSys/Data/credits.csv', encoding='latin-1', low_memory=False)
keywords = pd.read_csv('C:/RecoSys/Data/keywords.csv', encoding='latin-1', low_memory=False)

# Function to convert all non-integer IDs to NaN
def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan

# Clean the ids of df
movies['id'] = movies['id'].apply(clean_ids)

# Filter all rows that have a null ID
movies = movies[movies['id'].notnull()]

# Convert IDs into integer
movies['id'] = movies['id'].astype('int')
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')

# Print the first cast member of the first movie
movies.iloc[0]['crew'][0]

# Convert the string objects into the native python objects
from ast import literal_eval
features = ['cast', 'crew', 'keywords']
for feature in features:
    movies[feature] = movies[feature].apply(literal_eval)
    
# Print the first cast member of the first movie
movies.iloc[0]['crew'][0]

# Extract the direct's name. If director is not listed, return NaN
def get_director(x):
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return crew_member['name']
        return np.nan
    
# Define the new director feature
movies['director'] = movies['crew'].apply(get_director)

# Return the list top 3 elements
def generate_list(x):
    if isinstance(x, list):
        names = [item['name'] for item in x]
        # Check if more than 3 elements exist. If yes, return only first three
        # If not, return entire list
        if len(names) > 3:
            names = names[:3]
        return names
    # Return empty list in case of missing/malformed data
    return []

# Apply the generate_list function to cast and keywords
movies['cast'] = movies['cast'].apply(generate_list)
movies['keywords'] = movies['keywords'].apply(generate_list)

# Print the new features of the first 5 movies along with title
movies[['title', 'cast', 'director', 'keywords']].head(5)

# Removes spaces and converts to lowercase
def sanitize(x):
    if isinstance(x, list):
        # Strip spaces and convert to lowercase
        return [str.lower(i.replace(" ",""))for i in x]
    else:
        # Check if an item exists. If not, return empty string
        if isinstance (x, str):
            return str.lower(x.replace(" ",""))
        else:
            return ''
        
# Apply the sanitize function to cast, keywords, and director
for feature in ['cast', 'director', 'keywords']:
    movies[feature] = movies[feature].apply(sanitize)
    
# Function that creates a soup out of the desired metadata
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director']

# Create the new soup feature
movies['soup'] = movies.apply(create_soup, axis=1)

# Display the soup of the first movie
movies.iloc[0]['soup']

# Import CountVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import CountVectorizer

# Define a new CountVectorizer object and create vectors for the soup
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies['soup'])

# Cosine 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim = pd.DataFrame(cosine_sim, index=movies.index, columns=movies.index) # (46439, 46439)

# index-title을 뒤집는다
indices = pd.Series(movies.index, index=movies['title'])

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
