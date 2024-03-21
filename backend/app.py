import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from helpers.analysis import (tokenize, 
build_br_inverted_index, distinct_words, get_good_words, create_review_word_occurrence_matrix, 
compute_review_norms, build_wr_inverted_index, compute_idf)
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


ma_json_file_path = os.path.join(current_directory, 'de.json')
# cols = ['review_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text']
cols = ["review_id", "business_id", "stars_x", "text", "name", "address", "city", "state", "postal_code"]
with open(ma_json_file_path, 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data=data, columns=cols)
# print(df["text"])
distinct_words = distinct_words(tokenize, df) 
# print("distinct_words")
good_words = get_good_words(0.1, 0.9, df["text"], distinct_words)
# print(good_words)

# build inverted business-review index
br_inv_idx = build_br_inverted_index(df)
# print("br_inv_idx")

# build vector array of shape (review, good_words) - values are binary to start
# review index i is the same index it has in df
review_vectors = create_review_word_occurrence_matrix(tokenize, df, good_words)
# print(review_vectors)

# build word-review invertedd index. key = good type,
#value = list of tuples pertaining to review that has that good type
wr_inv_idx = build_wr_inverted_index(review_vectors, df, good_words)
print("final dict", wr_inv_idx)

#START cosine similarity computation
dummy_review = "this place is yummy and has good service. it is a restaurant that I will return to. chiara emory varsha teresa"
# replace dummy review with actuall user inputted review from frontend
input_review_dict = {"text": dummy_review}
input_review_df = pd.DataFrame([input_review_dict])
print("review df", input_review_df)

# vectorize review
input_review_vector = create_review_word_occurrence_matrix(tokenize, input_review_df, good_words)

print("review vectors shape", review_vectors.shape)
print("review vector shape", input_review_vector.shape)
print("review vector", input_review_vector)


idf = compute_idf(wr_inv_idx, len(df))
# print(idf)

doc_norms = compute_review_norms(wr_inv_idx, idf, len(df))
print(doc_norms)







def businesss_search(review, star_rating, zip_code):
   return df[:5] 

@app.route("/restaurants")
def restaurant_search():
   review = request.args.get("review")
   star_rating = request.args.get("star_rating")
   zip_code = request.args.get("zip_code")
   return businesss_search(review, star_rating, zip_code)
