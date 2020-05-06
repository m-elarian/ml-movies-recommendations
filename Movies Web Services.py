#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from ast import literal_eval

import warnings; warnings.simplefilter('ignore')


# In[2]:


credits = pd.read_csv("/home/mohamedelsayed/kaggle/input/Movies/credits.csv")


# In[3]:


keywords = pd.read_csv("/home/mohamedelsayed/kaggle/input/Movies/keywords.csv")


# In[4]:


movies = pd.read_csv("/home/mohamedelsayed/kaggle/input/Movies/movies_metadata.csv")


# In[5]:


indecies = movies[(movies.adult != 'True') & (movies.adult != 'False')].index
movies.drop(indecies, inplace = True)


# In[6]:


credits['id'] = credits['id'].astype('int')
keywords['id'] = keywords['id'].astype('int')
movies['id'] = movies['id'].astype('int')


# In[7]:


movies = movies.merge(credits,on='id')
movies = movies.merge(keywords,on='id')


# In[8]:


vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')


# In[9]:


m= vote_counts.quantile(0.95)


# In[10]:


vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('float')


# In[11]:


C= vote_averages.mean()


# In[12]:


movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[13]:


qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())][['title', 'imdb_id', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('float')
qualified['vote_average'] = qualified['vote_average'].astype('float')
qualified['popularity'] = qualified['popularity'].astype('float')


# In[14]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[15]:


qualified['score'] = qualified.apply(weighted_rating, axis=1)


# In[16]:


high_score = qualified.sort_values('score', ascending=False)


# In[17]:


high_popular = qualified.sort_values('popularity', ascending=False)


# In[18]:


links_small = pd.read_csv('/home/mohamedelsayed/kaggle/input/Movies/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[19]:


recommended_movies = movies[movies['id'].isin(links_small)]


# In[20]:


recommended_movies['tagline'] = recommended_movies['tagline'].fillna('')
recommended_movies['description'] = recommended_movies['overview'] + recommended_movies['tagline']
recommended_movies['description'] = recommended_movies['description'].fillna('')


# In[21]:


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(recommended_movies['description'])


# In[22]:


tfidf_matrix.shape


# In[23]:


cosine_sim_word = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[24]:


recommended_movies = recommended_movies.reset_index()
indices = pd.Series(recommended_movies.index, index=recommended_movies['title']).drop_duplicates()


# In[25]:


features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    recommended_movies[feature] = recommended_movies[feature].fillna('[]').apply(literal_eval)


# In[26]:


recommended_movies['genres'] = recommended_movies['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[27]:


recommended_movies['keywords'] = recommended_movies['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[28]:


s = recommended_movies.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s[:5]


# In[29]:


def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


# In[30]:


stemmer = SnowballStemmer('english')
recommended_movies['keywords'] = recommended_movies['keywords'].apply(filter_keywords)
recommended_movies['keywords'] = recommended_movies['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])


# In[31]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[32]:


recommended_movies['director'] = recommended_movies['crew'].apply(get_director)


# In[33]:


recommended_movies['main_actors'] = recommended_movies['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
recommended_movies['main_actors'] = recommended_movies['main_actors'].apply(lambda x: x[:3] if len(x) >=3 else x)


# In[34]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[35]:


# Apply clean_data function to your features.
features = ['main_actors', 'director', 'keywords', 'genres']

for feature in features:
    recommended_movies[feature] = recommended_movies[feature].apply(clean_data)


# In[36]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['main_actors']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

recommended_movies['soup'] = recommended_movies.apply(create_soup, axis=1)


# In[37]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(recommended_movies['soup'])


# In[38]:


# Compute the Cosine Similarity matrix based on the count_matrix
count_matrix = count_matrix.astype(np.float32)
cosine_sim_count = cosine_similarity(count_matrix, count_matrix)


# In[39]:


# Reset index of our main DataFrame and construct reverse mapping as before
recommended_movies = recommended_movies.reset_index()
indices = pd.Series(recommended_movies.index, index=recommended_movies['title'])


# In[40]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[41]:


links_small_id = pd.read_csv("/home/mohamedelsayed/kaggle/input/Movies/links_small.csv")[['movieId', 'tmdbId']]
links_small_id['tmdbId'] = links_small_id['tmdbId'].apply(convert_int)
links_small_id.columns = ['movieId', 'id']
links_small_id = links_small_id.merge(recommended_movies[['title', 'id']], on='id').set_index('title')


# In[42]:


indices_id = links_small_id.set_index('id')


# In[43]:


reader = Reader()
ratings = pd.read_csv("/home/mohamedelsayed/kaggle/input/Movies/ratings_small.csv")


# In[44]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# In[45]:


svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[46]:


trainset = data.build_full_trainset()
svd.fit(trainset)


# In[47]:


# Function that takes in movie title as input and outputs high rating movies
def get_high_rating_movies(movies_count):
    high_score = qualified.sort_values('score', ascending=False)

    # Return the top 10 most similar movies
    return high_score[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(movies_count)


# In[48]:


# Function that takes in movie title as input and outputs most popular movies
def get_popular_movies(movies_count):
    high_popular = qualified.sort_values('popularity', ascending=False)

    # Return the top 10 most similar movies
    return high_popular[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(movies_count)


# In[49]:


get_popular_movies(30)


# In[50]:


def get_best_movies_recommendations(title, movies_count, cosine_sim = cosine_sim_word):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:101]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = recommended_movies.iloc[movie_indices][['imdb_id', 'title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.0)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False)
    return qualified[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(movies_count)


# In[51]:


get_best_movies_recommendations('The Lion King', 5, cosine_sim_count)


# In[52]:


def get_user_recommendations(userId, title, movies_count, cosine_sim = cosine_sim_word):
    idx = indices[title]
    tmdbId = links_small_id.loc[title]['id']
    movie_id = links_small_id.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = recommended_movies.iloc[movie_indices][['imdb_id','title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_id.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(movies_count)


# In[53]:


get_user_recommendations(1, 'Avatar', 5, cosine_sim_count)


# In[54]:


get_user_recommendations(500, 'Avatar', 5, cosine_sim_count)

