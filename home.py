#import Pandas
import pandas as pd

#Simple Recommender

#load metadata
movies_metadata = pd.read_csv('/Users/rahuldas/Documents/archive/movies_metadata.csv',low_memory=False)

#print top 10
print(movies_metadata.head(10))

#calculate mean vote average
Mean = movies_metadata['vote_average'].mean()
print(Mean)

# Calculate the minimum number of votes required to be in the chart, min_votes
min_votes = movies_metadata['vote_count'].quantile(0.90)
print(min_votes)

# Filter out all qualified movies into a new DataFrame
q_movies = movies_metadata.copy().loc[movies_metadata['vote_count'] >= min_votes]
print(q_movies.shape)#number of rows and columns in this df

print(movies_metadata.shape)

# Function that computes the weighted rating of each movie
# x is my dataframe - q_movies47.61
def weighted_rating(x, m=min_votes, C=Mean):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))

#Content-Based Recommender - based on plot description

#Print plot overviews of the first 5 movies.
print(movies_metadata['overview'].head())

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Initiate count vectorizer. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
movies_metadata['overview'] = movies_metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(movies_metadata['overview'])

#Output the shape of tfidf_matrix
print(tfidf_matrix.shape)

#Array mapping from feature integer indices to feature name.
print(tfidf.get_feature_names()[1000:1010])

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cos_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print(cos_sim.shape)

print(cos_sim[1])

#Construct a reverse map of indices and movie titles
indices = pd.Series(movies_metadata.index, index=movies_metadata['title']).drop_duplicates()

print(indices[:2])


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cos_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]
    #print(idx)
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cos_sim[idx]))
    #print(sim_scores[:2])
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies_metadata['title'].iloc[movie_indices]

print(get_recommendations('Toy Story'))

