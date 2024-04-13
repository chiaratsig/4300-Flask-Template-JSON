import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from helpers.analysis import (tokenize, 
build_br_inverted_index, distinct_words, get_good_words, create_review_word_occurrence_matrix, 
compute_review_norms, build_wr_inverted_index, compute_idf, index_search, index_search2,
build_cr_inverted_index, create_top_category_vectors, create_query_vector, create_restaurant_vectors,
update_query_vector)
import pandas as pd
from itertools import repeat

############ TEMPLATE BEGIN ############

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
# with open(json_file_path, 'r') as file:
#     data = json.load(file)
#     episodes_df = pd.DataFrame(data['episodes'])
#     reviews_df = pd.DataFrame(data['reviews'])

# Sample search using json with pandas
# def json_search(query):
#     matches = []
#     merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
#     matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
#     matches_filtered = matches[['title', 'descr', 'imdb_rating']]
#     matches_filtered_json = matches_filtered.to_json(orient='records')
#     return matches_filtered_json

app = Flask(__name__)
CORS(app)

# @app.route("/")
# def home():
#     return render_template('base.html',title="sample html")

# @app.route("/episodes")
# def episodes_search():
#     text = request.args.get("title")
#     return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)

############ TEMPLATE END ############
    
########### P03 START ###########

# def business_search(review, star_rating, zip_code):
#     ma_json_file_path = os.path.join(current_directory, 'de.json')
#     # cols = ['review_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text']
#     cols = ["review_id", "business_id", "stars_x", "text", "name", "address", "city", "state", "postal_code"]
#     with open(ma_json_file_path, 'r') as file:
#         data = json.load(file)

#     df = pd.DataFrame(data=data, columns=cols)

#     distinct = distinct_words(tokenize, df) 

#     good_words = get_good_words(0.1, 0.9, df["text"], distinct)

#     # build inverted business-review index
#     br_inv_idx = build_br_inverted_index(df)

#     # build vector array of shape (review, good_words) - values are binary to start
#     # review index i is the same index it has in df
#     review_vectors = create_review_word_occurrence_matrix(tokenize, df, good_words)

#     # build word-review invertedd index. key = good type,
#     #value = list of tuples pertaining to review that has that good type
#     wr_inv_idx = build_wr_inverted_index(review_vectors, df, good_words)

#     #START cosine similarity computation
#     # dummy_review = "this place is yummy and has good service. it is a restaurant that I will return to. chiara emory varsha teresa"
#     # replace dummy review with actuall user inputted review from frontend
#     input_review_dict = {"text": review}
#     input_review_df = pd.DataFrame([input_review_dict])

#     # vectorize review
#     input_review_vector = create_review_word_occurrence_matrix(tokenize, input_review_df, good_words)

#     idf = compute_idf(wr_inv_idx, len(df))

#     doc_norms = compute_review_norms(wr_inv_idx, idf, len(df))

#     # dummy_rating = 1
#     returned_restaurants = index_search(input_review_df.iloc[0]["text"], wr_inv_idx, df, idf, doc_norms, int(star_rating))
#     return returned_restaurants

## CHIARA START P04
def business_search2():
    # Todo
    de_json_file_path = os.path.join(current_directory, 'de.json')
    # cols = ['review_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text']
    cols = ["review_id", "business_id", "stars_x", "text", "name", "address", "city", "state", "postal_code"]
    with open(de_json_file_path, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data=data, columns=cols)

    distinct = distinct_words(tokenize, df) 

    good_words = get_good_words(0.1, 0.9, df["text"], distinct)

    # build inverted business-review index
    br_inv_idx = build_br_inverted_index(df)

    # build vector array of shape (review, good_words) - values are binary to start
    # review index i is the same index it has in df
    review_vectors = create_review_word_occurrence_matrix(tokenize, df, good_words)
        

    # build word-review inverted index. key = good type,
    #value = list of tuples pertaining to review that has that good type
    wr_inv_idx = build_wr_inverted_index(review_vectors, df, good_words)

    # create dummy global category list
    dummy_top_categories = ['American', 'Gastropub', 'Chinese', 'Italian', 'Japanese', 'Czech', 'African',
                        'New', 'Old', 'Quick', 'Cheap', 'French', 'Old School', 'Local', 'Woman-Owned']

    # create dummy categories checked off by the user
    dummy_selected_categories = ['American', 'Gastropub', 'Cheap']

    # (GLOBAL) build an ar inverted index, where key = one of the top15 categories and the 
    #value is a single-linked list of review_ids pertaining to  reviews whos restaurants 
    #have that category
    #TODO: change build_cr_inverted_index to fit datastructure
    #TODO: swap out dummy_att_df for df
    dummy_cat_df = df.copy()
    dummy_cat_df['categories'] = [["American"]] * df.shape[0]

    cr_inv_idx = build_cr_inverted_index(dummy_cat_df, dummy_top_categories)

    # (GLOBAL) For each of the categories in the global categories list (n=15), create a combined 
    #review vector of all restaurants that have that category - AVERAGE each of the review vectors
    # initialize empty vectors
    top_category_vectors = create_top_category_vectors(review_vectors, cr_inv_idx,
                                                        dummy_top_categories, len(good_words))

    # (search-specific) For each of the attributes in the user-selected-checkboxes, 
    #ADD their vectors. This is the initial query (adding allows a restaurant with 3 of 
    #the desired selected attributes to likely be ranked higher than a restaurant with 
    #1 of the  desired selected attributes, for example
    initial_query = create_query_vector(dummy_top_categories, top_category_vectors,
                                        dummy_selected_categories, len(good_words))

    #START cosine similarity computation
    idf = compute_idf(wr_inv_idx, len(df))

    doc_norms = compute_review_norms(wr_inv_idx, idf, len(df))

    # Pass this initial query into an updated version of index_search in analysis.py 
    #(this initial query is the new value of input_review_vector in app.py). 
    #This will return returned_restaurants (n=5)
    initial_returned_restaurants = index_search2(good_words, initial_query, wr_inv_idx, df, idf, doc_norms)

    # Assign each of the returned_restaurants a rating [0,1] (dummy for now)
    dummy_scores = [0, 0.25, 0.5, 0.75, 1]

    # Build a vector to represent each of the 5 returned restaurants - average the review vector 
    #of each review pertaining to that restaurant
    restaurant_vectors = create_restaurant_vectors(review_vectors, initial_returned_restaurants, br_inv_idx, len(good_words))

    # For each of the 5 restaurants, add the (relevant restaurant vectors to the initial query vector)/(# relevant restaurants), 
    #subtract the (irrelevant restaurant vectors)/(#irrelevant restaurants)  from the initial query. 
    #This is the updated query
    updated_query = update_query_vector(initial_query, restaurant_vectors, dummy_scores)

    # Run business_search on this updated query
    updated_returned_restaurants = index_search2(good_words, updated_query, wr_inv_idx, df, idf, doc_norms)
    print("initial res", initial_returned_restaurants)
    print("updated res", updated_returned_restaurants)





business_search2()




#TODO: uncomment to route
# @app.route("/restaurants")
# def restaurant_search():
#     review = request.args.get("review")
#     star_rating = request.args.get("starRating")
#     zip_code = request.args.get("zipCode")
#     return business_search(review, star_rating, zip_code) 
   
########### P03 END ###########
    
de_json_file_path = os.path.join(current_directory, 'de.json')
cols = ["review_id", "business_id", "stars_x", "text", "name", "address", "city", "state", "postal_code"]

with open(de_json_file_path, 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data=data, columns=cols)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/tags")
def get_tags():
    tags = request.args.get("tags")
    pos = request.args.get("pos")
    tags = tags.strip().split(",")
    pos = pos.strip().split(",")
  
    print(pos)
    return pos

@app.route("/restaurantRatings")
def get_ratings():
    rating1 = request.args.get("rating1")
    rating2 = request.args.get("rating2")
    rating3 = request.args.get("rating3")
    rating4 = request.args.get("rating4")

  
    print(rating1, rating2, rating3, rating4)
    return rating1