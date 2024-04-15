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
    
de_json_file_path = os.path.join(current_directory, 'de.json')
cols = ["review_id", "business_id", "stars_x", "text", "name", "address", "city", "state", "postal_code", "categories"]

with open(de_json_file_path, 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data=data, columns=cols)
df["categories"] = df["categories"].astype("object")

name_row_dict = {}
for index, row in df.iterrows():
    name_row_dict[row["name"]] = index
    tempList = row["categories"].split(",")
    df.at[index, "categories"] = tempList



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

dummy_categories = ['American', 'Gastropub', 'Chinese', 'Italian', 'Japanese', 'Czech', 'African',
                        'New', 'Old', 'Quick', 'Cheap', 'French', 'Old School', 'Local', 'Woman-Owned']

# categories = ['American (Traditional)', 'Sandwiches', 'Breakfast & Brunch', 'Pizza', 'Fast Food', 'Mexican', 'Italian', 'Seafood', 'Coffee & Tea', 'Chinese', 'Japanese', 'Desserts', 'Mediterranean', 'Thai', 'Vegan', 'Vietnamese', 'Latin American', 'Indian', 'Middle Eastern', 'Korean']

# (GLOBAL) build an ar inverted index, where key = one of the top15 categories and the 
#value is a single-linked list of review_ids pertaining to  reviews whos restaurants 
#have that category
#TODO: change build_cr_inverted_index to fit datastructure
#TODO: swap out dummy_att_df for df
df = df.copy()
df['categories'] = [["American"]] * df.shape[0]

# cr_inv_idx = build_cr_inverted_index(df, categories)
cr_inv_idx = build_cr_inverted_index(df, dummy_categories)

# (GLOBAL) For each of the categories in the global categories list (n=15), create a combined 
#review vector of all restaurants that have that category - AVERAGE each of the review vectors
# initialize empty vectors
# category_vectors = create_top_category_vectors(review_vectors, cr_inv_idx,
#                                                         categories, len(good_words))
category_vectors = create_top_category_vectors(review_vectors, cr_inv_idx,
                                                         dummy_categories, len(good_words))

#START cosine similarity computation
idf = compute_idf(wr_inv_idx, len(df))

doc_norms = compute_review_norms(wr_inv_idx, idf, len(df))

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

# This route endpoint triggered when the user submits all tags that they like
@app.route("/state")
def get_state():
    state = request.args.get("state")
    print(state)
    return state

# variables that are created in an endpoint and needed in others
class Globals:
    # Global to keep track of the restaurants the user reviews
    reviewer_restaurants = []
    #Global to keep track of initial query
    initial_query = ""

# This route endpoint triggered when the user submits all tags that they like
@app.route("/tags")
def get_tags():
    selected_categories = request.args.get("tags")
    pos = request.args.get("pos")
    selected_categories = selected_categories.strip().split(",")
    pos = pos.strip().split(",")

    # Will take out when datasets have categories
    dummy_selected_categories = ['American', 'Gastropub', 'Cheap']

    # (search-specific) For each of the attributes in the user-selected-checkboxes, 
    #ADD their vectors. This is the initial query (adding allows a restaurant with 3 of 
    #the desired selected attributes to likely be ranked higher than a restaurant with 
    #1 of the  desired selected attributes, for example
    Globals.initial_query = create_query_vector(dummy_categories, category_vectors,
                                        dummy_selected_categories, len(good_words))
    # initial_query = create_query_vector(categories, category_vectors,
    #                                     selected_categories, len(good_words))


    # Pass this initial query into an updated version of index_search in analysis.py 
    #(this initial query is the new value of input_review_vector in app.py). 
    #This will return returned_restaurants (n=5)
    Globals.reviewer_restaurants = index_search2(good_words, Globals.initial_query, wr_inv_idx, df, idf, doc_norms)
    
    review_restaurants_info = []
    ### TODO: NEED A WAY TO GET THIS INFO OUT EASILY
    for restaurant in Globals.reviewer_restaurants:
        tup = []
        restaurant_row = name_row_dict[restaurant]
        tup.append(restaurant)
        tup.append(df["address"][restaurant_row])
        tup.append(df["city"][restaurant_row] + ", " + df["postal_code"][restaurant_row])
        tup.append("restaurant tags")
        # tup.append("restaurant tags")
        # tup.append("restaurant tags")
        review_restaurants_info.append(tuple(tup))

    print(review_restaurants_info)
    return review_restaurants_info
    # return {"review_restaurants_info": review_restaurants_info, "initial_query": initial_query, "reviewer_restaurants": reviewer_restaurants}

# This endpoint gets triggered when the user submits their restaurant ratings
@app.route("/restaurantRatings")
def get_ratings():
    rating1 = .1 * (int(request.args.get("rating1")) + 5)
    rating2 = .1 * (int(request.args.get("rating2")) + 5)
    rating3 = .1 * (int(request.args.get("rating3")) + 5)
    rating4 = .1 * (int(request.args.get("rating4")) + 5)
    rating5 = .1 * (int(request.args.get("rating5")) + 5)

    ratings = [rating1, rating2, rating3, rating4, rating5]
    # print(ratings)

    # Build a vector to represent each of the 5 returned restaurants - average the review vector 
    #of each review pertaining to that restaurant
    restaurant_vectors = create_restaurant_vectors(review_vectors, Globals.reviewer_restaurants, br_inv_idx, len(good_words))

    # For each of the 5 restaurants, add the (relevant restaurant vectors to the initial query vector)/(# relevant restaurants), 
    #subtract the (irrelevant restaurant vectors)/(#irrelevant restaurants)  from the initial query. 
    #This is the updated query
    updated_query = update_query_vector(Globals.initial_query, restaurant_vectors, ratings)

    # Run business_search on this updated query
    updated_returned_restaurants = index_search2(good_words, updated_query, wr_inv_idx, df, idf, doc_norms)
    print("initial res", Globals.reviewer_restaurants)
    print("updated res", updated_returned_restaurants)

    output_restaurants_info = []
    ### TODO: NEED A WAY TO GET THIS INFO OUT EASILY
    for restaurant in updated_returned_restaurants:
        tup = []
        restaurant_row = name_row_dict[restaurant]
        tup.append(restaurant)
        tup.append(df["address"][restaurant_row])
        tup.append(df["city"][restaurant_row] + ", " + df["postal_code"][restaurant_row])
        tup.append("restaurant tags")
        output_restaurants_info.append(tuple(tup))

    # restaurants_reccomended = [("Restaurant 1 Name", "Restaurant 1 Address", "Restaurant 1 Tags"), ("Restaurant 2 Name", "Restaurant 2 Address", "Restaurant 2 Tags"), ("Restaurant 3 Name", "Restaurant 3 Address", "Restaurant 3 Tags"), ("Restaurant 4 Name", "Restaurant 4 Address", "Restaurant 4 Tags"), ("Restaurant 5 Name", "Restaurant 5 Address", "Restaurant 5 Tags")]
    return output_restaurants_info
