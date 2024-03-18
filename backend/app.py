import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from helpers.analysis import (tokenize, num_dedup_tokens,
distinct_words, build_word_count, build_word_episode_distribution,
output_good_types, create_ranked_good_types, create_word_occurrence_matrix, create_weighted_word_freq_array)
import pandas as pd

############ TEMPLATE BEGIN ############

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    episodes_df = pd.DataFrame(data['episodes'])
    reviews_df = pd.DataFrame(data['reviews'])

# Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)

############ TEMPLATE END ############

ma_json_file_path = os.path.join(current_directory, 'ma_reviews.json')
cols = ['review_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text']
with open(ma_json_file_path, 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data=data, columns=cols)

distinct_words = distinct_words(tokenize, df) #uses a function in analysis.py to get a set of all distinct words among the reviews
print(distinct_words)

def businesss_search(review, star_rating, zip_code):
   return df[:5] 

@app.route("/restaurants")
def restaurant_search():
   review = request.args.get("review")
   star_rating = request.args.get("star_rating")
   zip_code = request.args.get("zip_code")
   return businesss_search(review, star_rating, zip_code)
