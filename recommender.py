import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# StreamLit initialization/Image
st.set_page_config(layout="wide")

st.title('Awesome Movie Recommender')

# Example image URL: https://liangfgithub.github.io/MovieImages/9.jpg?raw=true
image_url = 'https://liangfgithub.github.io/MovieImages/{}.jpg?raw=true'

movies_df = None
ratings_df = None
selected_movies_df = None
movie_titles = None
movie_list = None

genre_list = ["Suprise Me!",
              "Action", 
              "Adventure",
              "Animation",
              "Children's",
              "Comedy",
              "Crime",
              "Documentary",
              "Drama",
              "Fantasy",
              "Film-Noir",
              "Horror",
              "Musical",
              "Mystery",
              "Romance",
              "Sci-Fi",
              "Thriller",
              "War",
              "Western"]

my_userid = int(10000)
viewer_limit = 1000
ratings_limit = 3.9

@st.cache
def load_data():
    # Load and cache the pre-trained models
    movies_df = pd.read_csv('models/movies_df.csv', sep='\t', engine='python')
    ratings_df = pd.read_csv('models/ratings_df.csv', sep='\t', engine='python')
    selected_movies_df = pd.read_csv('models/selected_movies_df.csv', sep='\t', engine='python')

    # We'll randomize our initial movie selection to show different movies each time
    movie_list = selected_movies_df.sample(frac=1).reset_index(drop=True)
    movie_titles = dict(zip(movies_df['MovieID'], movies_df['Title']))
    
    return movies_df, ratings_df, selected_movies_df, movie_titles, movie_list


def recommend_by_genre(genre):
    if genre == "Suprise Me!":
        x = genre_list[random.randint(1, len(genre_list)-1)]
    else:
        x = genre
    genre_based_movies = movies_df[['MovieID', 'Title', x]]
    genre_based_movies = genre_based_movies[genre_based_movies[x] == 1]
    merged_genre_movies = pd.merge(ratings_df, genre_based_movies, how='inner', on='MovieID')
    
    high_rated_movies = merged_genre_movies.groupby(['MovieID']).agg({"Rating":"mean"})['Rating'].sort_values(ascending=False)
    high_rated_movies = high_rated_movies.to_frame()

    popular_movies_ingenre = merged_genre_movies.groupby(['MovieID']).agg({"Rating":"count"})['Rating'].sort_values(ascending=False)
    popular_movies_ingenre = popular_movies_ingenre.to_frame()
    
    popular_movies_ingenre.reset_index(level=0, inplace=True)
    popular_movies_ingenre.columns = ['MovieID', 'UserCount']

    highly_rated_popular_movies = pd.merge(high_rated_movies, popular_movies_ingenre, how = 'inner', on='MovieID')
    
    results = highly_rated_popular_movies[(highly_rated_popular_movies['UserCount']>viewer_limit) 
                                         & (highly_rated_popular_movies['Rating']>=ratings_limit)]
    
    results = results.sample(frac=1).reset_index(drop=True)
    return results["MovieID"].tolist()


def create_X(df):
    M = df['UserID'].nunique()
    N = df['MovieID'].nunique()

    user_mapper = dict(zip(np.unique(df["UserID"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["MovieID"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["UserID"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["MovieID"])))

    user_index = [user_mapper[i] for i in df['UserID']]
    item_index = [movie_mapper[i] for i in df['MovieID']]

    X = csr_matrix((df["Rating"], (user_index,item_index)), shape=(M,N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper


def predict_movies(df, rated_movies):
    results = []
    top_N = 50
    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(df)
    svd = TruncatedSVD(n_components=20, n_iter=10)
    Z = svd.fit_transform(X.T)
    pred = svd.inverse_transform(Z).T
    top_N_movies = pred[user_mapper[my_userid]].argsort()[-top_N:]
    
    # print(f"Top {top_N} Recommendations for UserId {my_userid}:")
    for i in top_N_movies:
        movie_id = movie_inv_mapper[i]
        # Remove movies already watched by user
        if not movie_id in rated_movies:
            results.append(movie_id)
            # st.write(movie_titles[movie_id])
    return results


def collab_selector(df):
    movie_selection = list()
    rated_movies = list()
    # movie_candidates = selected_movies_df.sample(frac=1).reset_index(drop=True)
    # Create 2x5 columns to display movies
    for i in range(2):
        cols = st.columns(5)
        for index, col in enumerate(cols):
            with col:
                title = movie_list.loc[5*i + index]['Title']
                movie_id = movies_df.loc[movies_df['Title'] == title, 'MovieID'].values[0]
                
                st.write(title)
                st.image('https://liangfgithub.github.io/MovieImages/{}.jpg?raw=true'.replace('{}', str(movie_id)))
                
                rating = st.slider('Your rating', 1, 5, key=int(movie_id))
                if (rating > 1):
                    selected = {'UserID' : my_userid, 'MovieID' : int(movie_id), 'Rating': rating}
                    movie_selection.append(selected)
                    rated_movies.append(int(movie_id))
    
    new_ratings_df = df.append(movie_selection)
    return new_ratings_df, rated_movies
    

def display_results1(results):
    # Create 2x5 columns to display movies
    for i in range(2):
        cols = st.columns(5)
        for index, col in enumerate(cols):
            with col:
                # Do pandas select by location - try it first!
                
                movie_id = results.loc[5*i + index]['MovieID']
                # st.write(movie_id)
                
                title = movies_df.loc[movies_df['MovieID'] == int(movie_id), 'Title'].values[0]
                st.write(title.ljust(60))
                
                # col.header(title)
                st.image('https://liangfgithub.github.io/MovieImages/{}.jpg?raw=true'.replace('{}', str(int(movie_id))))

                
def display_results2(results):
    # Create 2x5 columns to display movies
    for i in range(2):
        cols = st.columns(5)
        for index, col in enumerate(cols):
            with col:
                movie_id = results[5*i + index]
                title = movie_titles[movie_id]
                st.write(title.ljust(60))
                st.image(f'https://liangfgithub.github.io/MovieImages/{movie_id}.jpg?raw=true')
                

# Start the main loop
movies_df, ratings_df, selected_movies_df, movie_titles, movie_list = load_data()
# movie_list = selected_movies_df

st.markdown("""
    Use the menu at left to select type of recommender you like to use
""")

    
select_event =  add_selectbox = st.sidebar.selectbox(
    "Select the Recommender",
    ("System I: Genre-based", "System II: Collaborative Filtering")
)


if select_event == "System I: Genre-based":
    st.write("You selected System I: Genre-based Recommender")
             
    ### Process System I request
    option = None
    ### Add st.expander to offer a selection box
    sel_expander = st.expander(label='Expand to select a Movie Genre')
    with sel_expander:
        option = st.selectbox('Select the Genre you are interested in (or let us surprise you with our selection):', genre_list)
        # st.write('You selected:', option)
        # recommend = recommend_by_genre(option)
        # display_results(recommend)
        # st.write(rec)
    
    res_expander = st.expander(label='Expand to see your movie recommensations - ' + option)
    with res_expander:
        if option is not None:
            recommend = recommend_by_genre(option)
            display_results2(recommend)

elif select_event == "System II: Collaborative Filtering":
    st.write("You selected System II: Collaborative Filtering Recommender")    
    
    new_ratings_df = ratings_df.copy(deep=True)
    
    ### Process System II request
    sel_expander = st.expander(label='Expand to rate your movies')
    with sel_expander:          
        new_ratings_df, rated_movies = collab_selector(new_ratings_df)
    
    st.write('You have rated', len(new_ratings_df) - len(ratings_df), 'movies')
    # st.write(rated_movies)
    
    res_expander = st.expander(label='Expand to see your movie recommensations')
    with res_expander:
        if st.button('I am done with rating. Show me the movies'):            
            top_N_movies = predict_movies(new_ratings_df, rated_movies)
            display_results2(top_N_movies)
        # if option is not None:
            # recommend = recommend_by_genre(option)
            # display_results(recommend)
            # st.write('To be implemented')
    
else:
    st.write("First select an option") 


