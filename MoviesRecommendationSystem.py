#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)
# 
# * References
# 
# * https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system/notebook
# * https://www.kaggle.com/rounakbanik/movie-recommender-systems/notebook
# * https://www.kaggle.com/fabiendaniel/film-recommendation-engine/data
# 

# ## 1- Import libraries

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
from ast import literal_eval

import warnings; warnings.simplefilter('ignore')


# In[2]:


credits = pd.read_csv("/home/mohamedelsayed/kaggle/input/Movies/credits.csv")
credits.head(5)


# In[3]:


keywords = pd.read_csv("/home/mohamedelsayed/kaggle/input/Movies/keywords.csv")
keywords.head(5)


# In[4]:


movies = pd.read_csv("/home/mohamedelsayed/kaggle/input/Movies/movies_metadata.csv")
movies.head(5)


# In[5]:


indecies = movies[(movies.adult != 'True') & (movies.adult != 'False')].index
movies.drop(indecies, inplace = True)
print ("{} \n".format(movies['adult'].value_counts()))


# In[6]:


credits['id'] = credits['id'].astype('int')
keywords['id'] = keywords['id'].astype('int')
movies['id'] = movies['id'].astype('int')


# In[7]:


movies = movies.merge(credits,on='id')
movies = movies.merge(keywords,on='id')
movies.head(5)


# In[8]:


movies.columns


# In[9]:


movies.info()


# ## Demographic Filtering

# In[10]:


vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
vote_counts


# In[11]:


# m= movies_data['vote_count'].quantile(0.95)
m= vote_counts.quantile(0.95)
m


# In[12]:


vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('float')
vote_averages


# In[13]:


# C= movies_data['vote_average'].mean()
C= vote_averages.mean()
C


# In[14]:


movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
movies.head(10)


# In[15]:


qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())][['title', 'imdb_id', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('float')
qualified['vote_average'] = qualified['vote_average'].astype('float')
qualified['popularity'] = qualified['popularity'].astype('float')
qualified.shape


# In[16]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[17]:


qualified['score'] = qualified.apply(weighted_rating, axis=1)
qualified.head(10)


# In[18]:


pop = qualified.sort_values('popularity', ascending=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
plt.barh(pop['title'].head(20),pop['popularity'].head(20), align='center', color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")


# In[19]:


# Function that takes in movie title as input and outputs most similar movies
def get_popular_movies(movies_count):
    pop= qualified.sort_values('popularity', ascending=False)

    # Return the top 10 most similar movies
    return pop[['imdb_id','title']].head(movies_count)


# In[52]:


outF = open("top10movies.txt", "w")
for line in get_popular_movies(20):
  # write line to output file
  print("===")
  print(line)
  outF.write(line)
  outF.write("\n") 
    
outF.close()


# In[61]:



# Using readlines() 
populrmovies = open('populrmovies.txt', 'r') 
Lines = populrmovies.readlines() 
allMovies = ""  
count = 0
# Strips the newline character 
for line in Lines: 
    #print(line.strip()+",") 
    allMovies = allMovies + line.strip()+","
#print("Line{}: {}".format(count, line.strip())) 
print(allMovies)
allMovies = allMovies[:-1]
print(allMovies)


# In[62]:


# Using readlines() 
populrmovies = open('populrmovies.txt', 'r') 
Lines = populrmovies.readlines() 
populrMoviesLine = ""  
count = 0
# Strips the newline character 
for line in Lines: 
    #print(line.strip()+",") 
    populrMoviesLine = populrMoviesLine + line.strip()+","
    #print("Line{}: {}".format(count, line.strip())) 
print(populrMoviesLine)
populrMoviesLine = populrMoviesLine[:-1]
print(populrMoviesLine)


# In[20]:


get_popular_movies(30)


# ## Content Based Recommender

# In[21]:


links_small = pd.read_csv('/home/mohamedelsayed/kaggle/input/Movies/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
links_small.head(10)


# In[22]:


recommended_movies = movies[movies['id'].isin(links_small)]
recommended_movies.shape


# In[23]:


recommended_movies['tagline'] = recommended_movies['tagline'].fillna('')
recommended_movies['description'] = recommended_movies['overview'] + recommended_movies['tagline']
recommended_movies['description'] = recommended_movies['description'].fillna('')


# In[24]:


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(recommended_movies['description'])


# In[25]:


tfidf_matrix.shape


# In[26]:


cosine_sim_word = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim_word[0]


# In[27]:


recommended_movies = recommended_movies.reset_index()
indices = pd.Series(recommended_movies.index, index=recommended_movies['title']).drop_duplicates()


# In[28]:


def get_movie_recommendations(title, cosine_sim = cosine_sim_word):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return recommended_movies[['imdb_id','title']].iloc[movie_indices]


# In[29]:


get_movie_recommendations('Shutter Island')


# In[30]:


features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    recommended_movies[feature] = recommended_movies[feature].fillna('[]').apply(literal_eval)


# In[31]:


recommended_movies['genres'] = recommended_movies['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
print ("{} \n".format(recommended_movies['genres'].value_counts()))


# In[32]:


recommended_movies['keywords'] = recommended_movies['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
print ("{} \n".format(recommended_movies['keywords'].value_counts()))


# In[33]:


s = recommended_movies.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s[:5]


# In[34]:


def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


# In[35]:


stemmer = SnowballStemmer('english')
recommended_movies['keywords'] = recommended_movies['keywords'].apply(filter_keywords)
recommended_movies['keywords'] = recommended_movies['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
print ("{} \n".format(recommended_movies['keywords'].value_counts()))


# In[36]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[37]:


recommended_movies['director'] = recommended_movies['crew'].apply(get_director)
print ("{} \n".format(recommended_movies['director'].value_counts()))


# In[38]:


recommended_movies['main_actors'] = recommended_movies['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
recommended_movies['main_actors'] = recommended_movies['main_actors'].apply(lambda x: x[:3] if len(x) >=3 else x)
print ("{} \n".format(recommended_movies['main_actors'].value_counts()))


# In[39]:


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


# In[40]:


# Apply clean_data function to your features.
features = ['main_actors', 'director', 'keywords', 'genres']

for feature in features:
    recommended_movies[feature] = recommended_movies[feature].apply(clean_data)


# In[41]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['main_actors']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

recommended_movies['soup'] = recommended_movies.apply(create_soup, axis=1)


# In[42]:


recommended_movies[['imdb_id', 'title', 'genres', 'main_actors', 'director', 'keywords', 'soup']].head(5)


# In[43]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(recommended_movies['soup'])


# In[44]:


# Compute the Cosine Similarity matrix based on the count_matrix
count_matrix = count_matrix.astype(np.float32)
cosine_sim_count = cosine_similarity(count_matrix, count_matrix)


# In[45]:


# Reset index of our main DataFrame and construct reverse mapping as before
recommended_movies = recommended_movies.reset_index()
indices = pd.Series(recommended_movies.index, index=recommended_movies['title'])


# In[46]:


get_movie_recommendations('The Dark Knight Rises', cosine_sim_count)


# In[47]:


get_movie_recommendations('The Lion King', cosine_sim_count)


# In[48]:


outF = open("top10movies.txt", "w")
for line in get_popular_movies(20):
  # write line to output file
  outF.write(line)
  outF.write("\n") 
    
outF.close()


# ## Collaborative Filtering
