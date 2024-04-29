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
categories = [
              "Sandwiches",
              "Bars",
              "Pizza",
              "Nightlife",
              "Breakfast & Brunch",
              "Mexican",
              "Italian",
              "Coffee & Tea",
              "Fast Food",
              "Salad",
              "Burgers",
              "Delis",
              "Seafood",
              "Cafes",
              "Specialty Food",
            ]
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
    # ADD their vectors. This is the initial query (adding allows a restaurant with 3 of 
    # the desired selected attributes to likely be ranked higher than a restaurant with 
    # 1 of the  desired selected attributes
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
    
    # if(sum(Globals.initial_query) == 0):
    #     return "We're sorry, but there are no restaurants pertaining to your selected tags. Please choose a different set of tags."

    # print("initial query")
    # print(Globals.initial_query)
    # Pass this initial query into an updated version of index_search in analysis.py 
    # (this initial query is the new value of input_review_vector in app.py). 
    # This will return returned_restaurants (n=5)
    Globals.reviewer_restaurants = index_search2(good_words, Globals.initial_query, wr_inv_idx, df, idf, doc_norms)
    # print(Globals.reviewer_restaurants)
    review_restaurants_info = []
    for restaurant in Globals.reviewer_restaurants:
        tup = []
        restaurant = restaurant.lower()
        restaurant_rows = name_row_dict[restaurant]
        print(restaurant)
        print(restaurant_rows)
        tup.append(string.capwords(restaurant))
        tup.append(df["address"][restaurant_rows[0]] + ", " + df["city"][restaurant_rows[0]] + ", " + df["postal_code"][restaurant_rows[0]])
        tup.append("restaurant tags")

        reviews = []
        i = 0
        while i < 5 and i < len(restaurant_rows):
            reviews.append(df["text"][restaurant_rows[i]])
            i +=1 
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
        # chiara
        tup.append(df["address"][restaurant_rows[0]] + ", " + df["city"][restaurant_rows[0]] + ", " + df["postal_code"][restaurant_rows[0]])
        tup.append(df["city"][restaurant_rows[0]] + ", " + df["postal_code"][restaurant_rows[0]])
        
        attributes = df["attributes"][restaurant_rows[0]]
        # print("ATTRIBUTES", attributes)
        attr_str = ""
        # old attributes
        # if attributes != None:
        #     attr_keys = list(attributes.keys())
        #     i = 0
        #     while i < 5 and i < len(attr_keys): 
        #       if type(attributes[attr_keys[i]]) != dict:
        #         attr_str += attr_keys[i] + " - " + attributes[attr_keys[i]] + "<br>"
        #         i += 1

        # chiara attributes sunday 4/28
        if attributes != None:
            attr_keys = list(attributes.keys())
            print("ATTR KEYS", attr_keys)
            # atts = ['OutdoorSeating', 'DogsAllowed', 'WheelchairAccessible', 'RestaurantsDelivery']
            
            if 'OutdoorSeating' not in attr_keys or attributes['OutdoorSeating'] != 'True':
                attr_str += "Outdoor Seating: No <br>"
            else:
                attr_str += "Outdoor Seating: Yes <br>"

            if 'DogsAllowed' not in attr_keys or attributes['DogsAllowed'] != 'True':
                attr_str += "Dog Friendly: No <br>"
            else:
                attr_str += "Dog Friendly: Yes <br>"

            if 'WheelchairAccessible' not in attr_keys or attributes['WheelchairAccessible'] != 'True':
                attr_str += "Wheelchair Accessible: No <br>"
            else:
                attr_str += "Wheelchair Accessible: Yes <br>"

            if 'RestaurantsDelivery' not in attr_keys or attributes['RestaurantsDelivery'] != 'True':
                attr_str += "Delivery Available: No <br>"
            else:
                attr_str += "Delivery Available: Yes <br>"

    
        tup.append(attr_str)

        # restaurant_tags = {}
        # tags = ["WheelchairAccessible",  "WiFi", "OutdoorSeating", "BYOB"]
        # for review_id in restaurant_rows:
        #     if df["attributes"][review_id] != None:
        #         print(df["attributes"][review_id].keys())
        # print()

        tup.append("restaurant tags")
        output_restaurants_info.append(tuple(tup))

    # restaurants_reccomended = [("Restaurant 1 Name", "Restaurant 1 Address", "Restaurant 1 Tags"), ("Restaurant 2 Name", "Restaurant 2 Address", "Restaurant 2 Tags"), ("Restaurant 3 Name", "Restaurant 3 Address", "Restaurant 3 Tags"), ("Restaurant 4 Name", "Restaurant 4 Address", "Restaurant 4 Tags"), ("Restaurant 5 Name", "Restaurant 5 Address", "Restaurant 5 Tags")]
    return output_restaurants_info

# chiara testing

# test_state = "idaho"

# # get all needed inputs
# test_category_vectors = state_to_category_vectors[test_state]
# test_good_words = state_to_good_words[test_state]
# test_name_row_dict = state_to_name_row_dict[test_state]
# test_wr_inv_idx = state_to_wr_inv_idx[test_state]
# test_df = state_to_df[test_state]
# test_idf = state_to_idf[test_state]
# test_doc_norms = state_to_doc_norms[test_state]

# test_review_vectors = state_to_review_vectors[test_state]
# test_br_inv_idx = state_to_br_inv_idx[test_state]

# # make initial query
# test_initial_query = create_query_vector(categories, test_category_vectors,
#                                         [categories[0]], len(test_good_words))
# print("Test initial", test_initial_query)

# # get initial returned restaurants
# test_reviewer_restaurants = index_search2(test_good_words, test_initial_query, test_wr_inv_idx, test_df, test_idf, test_doc_norms)
# print("Test reviewer restaurants", test_reviewer_restaurants)

# # get intial returne restaurant vectors to pass into rocchio
# test_restaurant_vectors = create_restaurant_vectors(test_review_vectors, test_reviewer_restaurants, test_br_inv_idx, len(test_good_words))
# #print("rest vecrors", test_restaurant_vectors)

# # test 3 different updated results
# test_ratings0 = [0,0,0,0,0]
# test_ratings1 = [1,1,0,1,1]
# test_ratings5 = [0.25, 0.5, 0.25, 0.5, 0.25]
# test_updated_query0 = update_query_vector(test_initial_query, test_restaurant_vectors, test_ratings0)
# test_updated_query1 = update_query_vector(test_initial_query, test_restaurant_vectors, test_ratings1)
# test_updated_query5 = update_query_vector(test_initial_query, test_restaurant_vectors, test_ratings5)

# # get new set of returned restaurants for each set of updates
# test_updated_returned_restaurants0 = index_search2(test_good_words, test_updated_query0, test_wr_inv_idx, test_df, test_idf, test_doc_norms)
# test_updated_returned_restaurants1 = index_search2(test_good_words, test_updated_query1, test_wr_inv_idx, test_df, test_idf, test_doc_norms)
# test_updated_returned_restaurants5 = index_search2(test_good_words, test_updated_query5, test_wr_inv_idx, test_df, test_idf, test_doc_norms)

# # print new returned restaurants
# print("0 reviewer restaurants", test_updated_returned_restaurants0)
# print("1 reviewer restaurants", test_updated_returned_restaurants1)
# print("5 reviewer restaurants", test_updated_returned_restaurants5)
