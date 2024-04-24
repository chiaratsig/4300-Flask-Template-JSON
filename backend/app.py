import json
import os
import string
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import state_processing
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from helpers.analysis import (index_search2, create_query_vector, create_restaurant_vectors,
update_query_vector)
import pandas as pd
from itertools import repeat

############ TEMPLATE BEGIN ############

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)

# Specify the path to the JSON file relative to the current script
# json_file_path = os.path.join(current_directory, '/state_data')
# print(json_file_path)

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

# variables that are created in an endpoint and needed in others
class Globals:
    # Global to keep track of the restaurants the user reviews
    reviewer_restaurants = []
    #Global to keep track of initial query
    initial_query = ""
    #Global to keep track of state
    state = ""

state_to_df, state_to_idf, state_to_good_words, state_to_wr_inv_idx, state_to_br_inv_idx, state_to_name_row_dict, state_to_doc_norms, state_to_review_vectors, state_to_category_vectors = state_processing.data_processing(current_directory)
print(state_to_category_vectors.keys())
categories = ['American (Traditional)', 'Sandwiches', 'Breakfast & Brunch', 'Pizza', 'Fast Food', 'Mexican', 'Italian', 'Seafood', 'Coffee & Tea', 'Chinese', 'Japanese', 'Desserts', 'Mediterranean', 'Thai', 'Vegan', 'Vietnamese', 'Latin American', 'Indian', 'Middle Eastern', 'Korean']
categories = list(map(lambda x:x.lower(), categories))

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

# This route endpoint triggered when the user submits all tags that they like
@app.route("/state")
def get_state():
    Globals.state = request.args.get("state")
    print(Globals.state)
    # print(state_to_df[Globals.state]["text"])
    return Globals.state

# This route endpoint triggered when the user submits all tags that they like
@app.route("/tags")
def get_tags():
    # categories = request.args.get("tags")
    # categories = categories.strip().split(",")

    selected_categories = request.args.get("pos")
    selected_categories = selected_categories.strip().split(",")

    print("selected categories")
    # print(selected_categories)

    # Will take out when datasets have categories
    # dummy_selected_categories = ['American', 'Gastropub', 'Cheap']

    # (search-specific) For each of the attributes in the user-selected-checkboxes, 
    #ADD their vectors. This is the initial query (adding allows a restaurant with 3 of 
    #the desired selected attributes to likely be ranked higher than a restaurant with 
    #1 of the  desired selected attributes, for example
    # Globals.initial_query = create_query_vector(dummy_categories, category_vectors,
    #                                     dummy_selected_categories, len(good_words))   
    category_vectors = state_to_category_vectors[Globals.state]
    good_words = state_to_good_words[Globals.state]
    name_row_dict = state_to_name_row_dict[Globals.state]
    wr_inv_idx = state_to_wr_inv_idx[Globals.state]
    df = state_to_df[Globals.state]
    idf = state_to_idf[Globals.state]
    doc_norms = state_to_doc_norms[Globals.state]

    print(df["name"])

    Globals.initial_query = create_query_vector(categories, category_vectors,
                                        selected_categories, len(good_words))

    # print("initial query")
    # print(Globals.initial_query)
    # Pass this initial query into an updated version of index_search in analysis.py 
    #(this initial query is the new value of input_review_vector in app.py). 
    #This will return returned_restaurants (n=5)
    Globals.reviewer_restaurants = index_search2(good_words, Globals.initial_query, wr_inv_idx, df, idf, doc_norms)
    # print(Globals.reviewer_restaurants)
    review_restaurants_info = []
    for restaurant in Globals.reviewer_restaurants:
        print()
        print()
        print(restaurant)
        print("HERE")
        print()
        tup = []
        restaurant = restaurant.lower()
        restaurant_rows = name_row_dict[restaurant]
        tup.append(string.capwords(restaurant))
        tup.append(df["address"][restaurant_rows[0]] + df["city"][restaurant_rows[0]] + ", " + df["postal_code"][restaurant_rows[0]])
        tup.append("restaurant tags")

        reviews = []
        i = 0
        while i < 5 and i < len(restaurant_rows):
            reviews.append(df["text"][i])
            i +=1 
        print(reviews)
        tup.append(reviews)
        review_restaurants_info.append(tuple(tup))


    # print(review_restaurants_info)
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
    review_vectors = state_to_review_vectors[Globals.state]
    good_words = state_to_good_words[Globals.state]
    wr_inv_idx = state_to_wr_inv_idx[Globals.state]
    df = state_to_df[Globals.state]
    idf = state_to_idf[Globals.state]
    br_inv_idx = state_to_br_inv_idx[Globals.state]
    doc_norms = state_to_doc_norms[Globals.state]
    name_row_dict = state_to_name_row_dict[Globals.state]

    # Build a vector to represent each of the 5 returned restaurants - average the review vector 
    #of each review pertaining to that restaurant
    restaurant_vectors = create_restaurant_vectors(review_vectors, Globals.reviewer_restaurants, br_inv_idx, len(good_words))

    # For each of the 5 restaurants, add the (relevant restaurant vectors to the initial query vector)/(# relevant restaurants), 
    #subtract the (irrelevant restaurant vectors)/(#irrelevant restaurants)  from the initial query. 
    #This is the updated query
    updated_query = update_query_vector(Globals.initial_query, restaurant_vectors, ratings)

    # Run business_search on this updated query
    updated_returned_restaurants = index_search2(good_words, updated_query, wr_inv_idx, df, idf, doc_norms)
    # print("initial res", Globals.reviewer_restaurants)
    # print("updated res", updated_returned_restaurants)

    output_restaurants_info = []
    for restaurant in updated_returned_restaurants:
        tup = []
        restaurant = restaurant.lower()
        restaurant_rows = name_row_dict[restaurant]
        tup.append(string.capwords(restaurant))
        tup.append(df["address"][restaurant_rows[0]])
        tup.append(df["city"][restaurant_rows[0]] + ", " + df["postal_code"][restaurant_rows[0]])
        tup.append("restaurant tags")
        output_restaurants_info.append(tuple(tup))

    # restaurants_reccomended = [("Restaurant 1 Name", "Restaurant 1 Address", "Restaurant 1 Tags"), ("Restaurant 2 Name", "Restaurant 2 Address", "Restaurant 2 Tags"), ("Restaurant 3 Name", "Restaurant 3 Address", "Restaurant 3 Tags"), ("Restaurant 4 Name", "Restaurant 4 Address", "Restaurant 4 Tags"), ("Restaurant 5 Name", "Restaurant 5 Address", "Restaurant 5 Tags")]
    return output_restaurants_info
