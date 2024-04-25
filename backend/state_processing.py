import os
import helpers.analysis as helpers
# import app
import json
import pandas as pd

def data_processing(current_directory):
# Get the directory of the current script
  state_to_json = {
      "alabama" : "ab.json",
      "arizona" : "az.json",
      "california" : "ca.json",
      "delaware" : "de.json", 
      "florida" : "fl.json",
      "idaho" : "id.json",
      "illinois" : "il.json",
      "indiana" : "in.json",
      "louisiana" : "la.json",
      "missouri" : "mo.json",
      "new jersey" : "nj.json",
      "nevada" : "nv.json",
      "pennsylvania" : "pa.json",
      "tennessee" : "tn.json",
  }

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
              "Speciality Food",
            ]  
  categories = list(map(lambda x:x.lower(), categories))

  state_to_df = {}
  state_to_idf = {}
  state_to_good_words = {}
  state_to_wr_inv_idx = {}
  state_to_br_inv_idx = {}
  state_to_name_row_dict = {}
  state_to_doc_norms = {}
  state_to_review_vectors = {}
  state_to_category_vectors = {}
  # state_to_restaurant_to_reviews = {}

  for state in state_to_json.keys():
    # ROOT_PATH for linking with all your files. 
    # Feel free to use a config.py or settings.py with a global export variable
    os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))
     
    # Get the directory of the current script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # print("cur dir")
    # print(current_directory)

    json_file_path = os.path.join(current_directory, "state_data", state_to_json[state])
    # print(json_file_path)
    cols = ["review_id", "business_id", "stars_x", "text", "name", "address", "city", "state", "postal_code", "categories", "attributes"]

    with open(json_file_path, 'r') as file:
      data = json.load(file)

    df = pd.DataFrame(data=data, columns=cols)
    df["categories"] = df["categories"].astype("object")
    df["attributes"] = df["attributes"].astype("object")
    state_to_df[state] = df

    name_row_dict = {}
    for index, row in df.iterrows():
      r_name = row["name"].lower()
      if r_name not in name_row_dict.keys():
        name_row_dict[r_name] = []
      name_row_dict[row["name"].lower()].append(index)

      tempCategories = row["categories"].split(",")
      tempCategories = list(map(lambda x:x.lower(), tempCategories))
      df.at[index, "categories"] = tempCategories

    state_to_name_row_dict[state] = name_row_dict


    distinct = helpers.distinct_words(helpers.tokenize, df) 
    good_words = helpers.get_good_words(0.2, 0.8, df["text"], distinct)
    state_to_good_words[state] = good_words

    # build inverted business-review index
    br_inv_idx = helpers.build_br_inverted_index(df)
    state_to_br_inv_idx[state] = br_inv_idx

    # build vector array of shape (review, good_words) - values are binary to start
    # review index i is the same index it has in df
    review_vectors = helpers.create_review_word_occurrence_matrix(helpers.tokenize, df, good_words)  
    state_to_review_vectors[state] = review_vectors

    # build word-review inverted index. key = good type,
    #value = list of tuples pertaining to review that has that good type
    wr_inv_idx = helpers.build_wr_inverted_index(review_vectors, df, good_words)
    state_to_wr_inv_idx[state] = wr_inv_idx

    # (GLOBAL) build an ar inverted index, where key = one of the top15 categories and the 
    #value is a single-linked list of review_ids pertaining to  reviews whos restaurants 
    #have that category
    cr_inv_idx = helpers.build_cr_inverted_index(df, categories)

    # (GLOBAL) For each of the categories in the global categories list (n=15), create a combined 
    #review vector of all restaurants that have that category - AVERAGE each of the review vectors
    # initialize empty vectors
    category_vectors = helpers.create_top_category_vectors(review_vectors, cr_inv_idx,
                                                            categories, len(good_words))
    state_to_category_vectors[state] = category_vectors

    #START cosine similarity computation
    idf = helpers.compute_idf(wr_inv_idx, len(df))
    state_to_idf[state] = idf

    helpers.doc_norms = helpers.compute_review_norms(wr_inv_idx, idf, len(df))
    doc_norms = helpers.compute_review_norms(wr_inv_idx, idf, len(df))
    state_to_doc_norms[state] = doc_norms

  return state_to_df, state_to_idf, state_to_good_words, state_to_wr_inv_idx, state_to_br_inv_idx, state_to_name_row_dict, state_to_doc_norms, state_to_review_vectors, state_to_category_vectors