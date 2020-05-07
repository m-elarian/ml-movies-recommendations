from flask import Flask
from flask_restful import Resource, Api
from flask import request

import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import pickle
# load numpy array from csv file
from numpy import loadtxt
# load numpy array from npy file
from numpy import load


#import warnings; warnings.simplefilter('ignore')

app = Flask(__name__)
app.config["DEBUG"] = True
api = Api(app)

#============================================================================================
#============================================================================================
@app.route('/',methods=["GET","POST"])
def hello_world():
    return 'movie recommendations'
	
@app.route('/popularMovies', methods=['GET'])
def home():
    if 'movies_count' in request.args:
        movies_count= request.args['movies_count']
        print (movies_count)
    else:
        movies_count = 10
		
    qualified = pd.read_csv("qualified.csv")		
    # Function that takes in movie count as input and outputs high rating movies
    def get_high_rating_movies(movies_count):
       high_score = qualified.sort_values('popularity', ascending=False)
       return high_score[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(movies_count)

    Lines =get_high_rating_movies(10)['imdb_id']
    returnLine = ""   
    # Strips the newline character 
    for line in Lines: 
        #print(line.strip()+",") 
        returnLine = returnLine + line.strip()+"," 
    print(returnLine)
    returnLine = returnLine[:-1]
    print(returnLine)
    return returnLine	   
 
#============================================================================================
#============================================================================================

@app.route('/getHighRatingMovies', methods=['GET'])
def getHighRatingMovies(): 
    if 'movies_count' in request.args:
        movies_count= request.args['movies_count']
        print (movies_count)
    else:
        movies_count = 10
		
    qualified = pd.read_csv("qualified.csv")		
    # Function that takes in movie count as input and outputs high rating movies
    def get_high_rating_movies(movies_count):
       high_score = qualified.sort_values('score', ascending=False)
       return high_score[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(movies_count)

    Lines =get_high_rating_movies(10)['imdb_id']
    returnLine = ""   
    # Strips the newline character 
    for line in Lines: 
        #print(line.strip()+",") 
        returnLine = returnLine + line.strip()+"," 
    print(returnLine)
    returnLine = returnLine[:-1]
    print(returnLine)
    return returnLine	   

#============================================================================================
#========================= bestMoviesRecommendations =========================
#============================================================================================

@app.route('/bestMoviesRecommendations', methods=['GET'])
def bestMoviesRecommendations():
    # (title, cosine_sim = cosine_sim_word):
    if 'imdb_id' in request.args:
        imdb_id= request.args['imdb_id']
        print (imdb_id)
    else:
        return 'please choose imdb_id. '
    print (imdb_id)

    if 'movies_count' in request.args:
        movies_count= request.args['movies_count']
        print (movies_count)
    else:
        movies_count = 10
    recommended_movies = pd.read_csv("./recommended_movies.csv")

    
    # load array
    #cosine_sim_word = load('cosine_sim_word.npy')
    # load numpy array from npy file
    from numpy import load
    # load array
    cosine_sim_count = load('cosine_sim_count.npz')
    cosine_sim_count = cosine_sim_count['arr_0']
	
        # Function that takes in movie imdb_id as input and outputs similar high rated movies based on metadata
		
    def get_best_movies_metadata_recommendations(imdb_id, count):
        recommended = recommended_movies.reset_index()
        indices_mov_mtdt = pd.Series(recommended.index, index=recommended['imdb_id'])
        idx = indices_mov_mtdt[imdb_id]
        sim_scores = list(enumerate(cosine_sim_count[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:101]
        movie_indices = [i[0] for i in sim_scores]
        
        movies = recommended.iloc[movie_indices][['imdb_id', 'title', 'vote_count', 'vote_average', 'year']]
        vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(0.0)
        def weighted_rating(x):
            v = x['vote_count']
            R = x['vote_average']
            return (v/(v+m) * R) + (m/(m+v) * C)
        qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')
        qualified['wr'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.sort_values('wr', ascending=False)
        return qualified[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(count)
    
    Lines = get_best_movies_metadata_recommendations(imdb_id, movies_count )['imdb_id']
    returnLine = ""   
    # Strips the newline character 
    for line in Lines: 
        #print(line.strip()+",") 
        returnLine = returnLine + line.strip()+"," 
    print(returnLine)
    returnLine = returnLine[:-1]
    print(returnLine)
    return returnLine
	
#============================================================================================
#========================= getUserRecommendations =========================
#============================================================================================
@app.route('/getUserRecommendations', methods=['GET'])
def getUserRecommendations():

    #get parameters userId, title, movies_count
    if 'movieUserId' in request.args:
        movieUserId=  int(request.args['movieUserId'])
        print (movieUserId)
    else:
        return 'please choose movieUserId. '

    if 'imdb_id' in request.args:
        imdb_id= request.args['imdb_id']
        print (imdb_id)
    else:
        return 'please choose imdb_id. '

    if 'movies_count' in request.args:
        movies_count= request.args['movies_count']
        print (movies_count)
    else:
        movies_count = 10
		
    # load array
    #cosine_sim_word = load('cosine_sim_word.npy')
    cosine_sim_count = load('cosine_sim_count.npz')
    cosine_sim_count = cosine_sim_count['arr_0']	


    # load array
    recommended_movies =pd.read_csv('recommended_movies.csv')
    
    svd=pickle.load(open('svdmodel.pkl','rb'))
    
    links_small_id = pd.read_csv("links_small_id.csv")
 
    def convert_int(x):
        try:
            return int(x)
        except:
            return np.nan
     
    # Function that takes in movie imdb_id as input and outputs similar movies based on user prediction
    def get_user_recommendations(userNo, imdb_id, movies_count):
        indices_user_id = links_small_id.set_index('id')
        links_small = links_small_id.reset_index()
        indices_mov_mtdt = pd.Series(links_small.index, index=links_small['imdb_id'])
    
        idx = indices_mov_mtdt[imdb_id]
        sim_scores = list(enumerate(cosine_sim_count[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]
        movie_indices = [i[0] for i in sim_scores]
        
        movies = recommended_movies.iloc[movie_indices][['imdb_id','title', 'vote_count', 'vote_average', 'year', 'id']]
        print(userNo)
        print(userNo)
        movies['est'] = movies['id'].apply(lambda x: svd.predict(userNo, indices_user_id.loc[x]['movieId']).est)
        movies = movies.sort_values('est', ascending=False)
        return movies[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(movies_count)
    print(movieUserId)
    Lines = get_user_recommendations(movieUserId, imdb_id, movies_count)['imdb_id']
    returnLine = ""  
    # Strips the newline character 
    for line in Lines: 
        #print(line.strip()+",") 
        returnLine = returnLine + line.strip()+"," 
    print(returnLine)
    returnLine = returnLine[:-1]
    print(returnLine)
	
    return returnLine
	
#============================================================================================
#============================================================================================	
if __name__ == '__main__':
   app.run(debug=True)
