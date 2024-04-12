from typing import List, Tuple, Dict
from collections.abc import Callable
import numpy as np
import pandas as pd
import re
from itertools import repeat

def tokenize(text: str) -> List[str]:
    """Returns a list of words that make up the text.
    
    Note: for simplicity, lowercase everything.
    Requirement: Use Regex to satisfy this function
    
    Parameters
    ----------
    text : str
        The input string to be tokenized.

    Returns
    -------
    List[str]
        A list of strings representing the words in the text.
    """
    return re.findall('[a-z]+', text.lower())

def distinct_words(tokenize_method: Callable[[str], List[str]],
    input_reviews: List[Tuple[str, List[Dict[str, str]]]]) -> int:
    """Returns the set of distinct tokens used in an entire transcript

    Parameters
    ----------
    tokenize_method : Callable[[str], List[str]]
        A method to tokenize a string into a list of strings representing words.
    tokenize_transcript_method: Callable[[Callable[[str], List[str]], Tuple[str, List[Dict[str, str]]]], List[str]]
        A method that returns a list of tokens contained in an entire transcript, 
        given a tokenization method and a transcript.
    input_transcripts : List[Tuple[str, List[Dict[str, str]]]]
        A list of tuples containing a transcript UID mapped to a list of utterances 
        (specified as a dictionary with ``speaker``, ``text``, and ``timestamp`` as keys).

    Returns
    -------
    set
        The set of distinct tokens in all the transcripts.
    """
    tokens = []
    for review_idx in input_reviews.index:
      tokens += tokenize_method(input_reviews["text"][review_idx])
    return set(tokens)

def get_good_words(min_percentage, max_percentage, reviews, distinct_words):
  """Returns the set of distinct tokens used in a percentage range of review

  Parameters
  ----------
  min_percentage : int
      The minimum percentage of reviews a word must be in to be a good word.
  max_percentage: int
      The minimum percentage of reviews a word must be in to be a good word.
  reviews : DataFrame
      A column of a DataFrame with the text of all reviews

  Returns
  -------
  good_words
    The set of distinct good words among all the reviews.
  """
  words = {}
  for review in reviews:
    tokens = tokenize(review)
    review_words = {}
    for token in tokens:
       if token not in review_words:
        review_words[token] = True
        if token not in words.keys():
            words[token] = 0
        words[token] += 1
  good_words = []

  for key in words.keys():
    word_percentage = words[key] / len(reviews)
    if word_percentage >= min_percentage and word_percentage <= max_percentage:
      good_words.append(key)
  return sorted(good_words)

def output_good_types(input_word_counts: Dict[str, str]) -> List[str]:
    """Returns a list of good types in alphabetically sorted order

    Parameters
    ----------
    input_word_counts : Dict[str, str]
        A dictionary of word mappeds to the number of episodes they appear in.

    Returns
    -------
    List[str]
        A list of all the words that appear in more than one episode, sorted alphabetically.
    """
    good_words = []
    for key in input_word_counts.keys():
      if input_word_counts[key] > 1:
        good_words.append(key)
    return sorted(good_words)

# different from the one in a4
def build_br_inverted_index(df: List[dict]) -> dict:
    """Builds an inverted index from the reviews. 

    Arguments
    =========

    df: DataFrame.
        Each review has a corresponding business_id.

    Returns
    =======

    inverted_index: dict
        Key = business_id, value = single-linked list of tuples
        pertaining to the business.
        Tuple[0] is the index of the review row in the df,
        tuple[1] is the review_ids

    """
    inv_idx = {}
    for i in range(df.shape[0]):
        if df.iloc[i]['business_id'] not in inv_idx.keys():
          inv_idx[df.iloc[i]['business_id']] = list((i, df.iloc[i]['review_id']))
        else:
          inv_idx[df.iloc[i]['business_id']].append((i, df.iloc[i]['review_id']))

    return inv_idx

# different from the one in a4/above
def create_review_word_occurrence_matrix(
    tokenize_method: Callable[[str], List[str]],
    input_df: List[str],
    input_good_types: List[str]) -> np.ndarray:
    """Returns a numpy array of shape n_reviews by n_good_types such that the 
    entry (ij) indicates if review i contains word j (binary).

    Parameters
    ----------
    tokenize_method : Callable[[str], List[str]]
        A method to tokenize a string into a list of strings representing words.
    input_df: List[str]
        df where each row pertains to a review
    input_good_types: List[str]
        A list of all good types, sorted alphabetically.

    Returns
    -------
    np.ndarray
        A numpy array of shape n_reviews by n_good_types such that the 
        entry (ij) indicates how often speaker i says word j.
    """
    word_occurence_matrix = np.zeros((input_df.shape[0], len(input_good_types)))
    good_types = np.array(input_good_types)

    for i in range(input_df.shape[0]):
       tokens = tokenize_method(input_df.iloc[i]['text'])
       for token in tokens:
          if token in input_good_types:
             word_occurence_matrix[i][input_good_types.index(token)] += 1
    return word_occurence_matrix


# different from the one in a4
def build_wr_inverted_index(
    review_vectors: np.ndarray,
    input_df: pd.DataFrame,
    input_good_types: List[str]) -> dict:
    """Builds an inverted index from the review vectors. 

    Arguments
    =========
    review_vectors : array
        Vectorized review, with shape (n_reviews, n_good_types)
    input_df: DataFrame.
        DataFrame that maps where each row has a unique review_id.
    input_good_types: List[str]
        list of good types

    Returns
    =======

    inv_idx: dict
        Key = good type (str), value = single-linked list of tuples
        pertaining to the reviews that contain that type.
        Tuple[0] is the index of the review row in the df,
        tuple[1] is the review_id
        tuple[2] is the frequency of the word in the review = review_vectors[i][j]

    """
    inv_idx = dict((word, []) for word in input_good_types)

    for i in range(review_vectors.shape[0]):
       review_index = i
       review_id = input_df.iloc[i]["review_id"]
       for j in range(len(input_good_types)):
          word = input_good_types[j]
          freq = review_vectors[i][j]
          
          if freq != 0:
            inv_idx[word].append((review_index, review_id, freq))
    return inv_idx

# def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
def compute_idf(inv_idx, n_docs):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    Hint: Make sure to use log base 2.

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """
    idf = dict()

    for key in inv_idx.keys():
        # docs = inv_idx[key]
    #   if len(docs) >= min_df:         
        idf_t = np.log2((n_docs)/(len(inv_idx[key]) + 1))
        # df_ratio = len(inv_idx[key])/n_docs

        # if df_ratio <= max_df_ratio:
        idf[key] = idf_t
   
    return idf

def compute_review_norms(index, idf, n_reviews):
    """Precompute the euclidean norm of each document.
    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.
    norms: np.array, size: n_reviews
        norms[i] = the norm of review i.
    """

    # TODO-6.1
    norms = np.zeros((n_reviews))

    for key in index.keys():
      tf_revs = index[key]
      for (review, review_id, tf) in tf_revs:
        if key in idf.keys():
          norms[review] += (tf * idf[key]) ** 2

    return np.sqrt(norms)

def accumulate_dot_scores(query_word_counts: dict, index: dict, idf: dict) -> dict:
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Arguments
    =========

    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.
    Returns
    =======

    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    # TODO-7.1
    dot_scores = dict()

    for key in query_word_counts.keys():
      q_i = query_word_counts[key]
      if key in index.keys():
        for (review, review_id, tf) in index[key]:
          if review not in dot_scores.keys():
            dot_scores[review] = q_i * tf * idf[key] * idf[key]
          else:
            dot_scores[review] += q_i * tf

    return dot_scores
       
       
def index_search(
    query: str,
    index: dict,
    df: pd.DataFrame,
    idf,
    doc_norms,
    star_rating,
    score_func=accumulate_dot_scores,
    tokenizer=tokenize,
) -> List[Tuple[int, int]]:
    """Search the collection of documents for the given query

    Arguments
    =========

    query: string,
        The query we are looking for.

    index: an inverted index as above, key = word, value = list of reviews

    df: dataframe mapping review to business name

    idf: idf values precomputed as above

    doc_norms: document norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See Q7)

    tokenizer: a TreebankWordTokenizer

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.

    Note:

    """

    # TODO-8.1
    results = []

    query = query.lower()
    tokens = tokenizer(query)

    tf = dict()
    for token in tokens:
      if token not in tf.keys():
        tf[token] = 1
      else:
        tf[token] += 1

    # compute query norm
    query_norm = 0
    for key in tf.keys():
      if key in idf.keys():
        query_norm += (tf[key] * idf[key]) ** 2
    query_norm = np.sqrt(query_norm)
    
    cossim_numerator = score_func(tf, index, idf)
    cossim_denominator = query_norm * doc_norms


    for doc in cossim_numerator.keys():
      results.append((cossim_numerator[doc] / cossim_denominator[doc], doc))

    results.sort(key=lambda x: x[0], reverse=True)

    
    # based on star review, either grab first 5 or last 5 reviews
    if star_rating >=3:
       returned_5_reviews = results[:5]
    else:
       returned_5_reviews = results[-5:]
    
    returned_5_restaurants = [df.iloc[x[1]]['name'] for x in returned_5_reviews]

    return returned_5_restaurants


def build_cr_inverted_index(df: List[dict],
                            top_categories: List[str]) -> dict:
    """Builds an inverted index from the reviews. 

    Arguments
    =========

    df: DataFrame.
        Each review has a corresponding list of categories pertaining
        to the restaurant.

    top_categories: List.
        List of top 15 categories user can check-off

    Returns
    =======

    inverted_index: dict
        Key = review_id, value = single-linked list of review_ids

    """
    values = [[], [], [], [], [], [], [], [], [], [], 
              [], [], [], [], []]
    inv_idx = dict(zip(top_categories, values))
    for i in range(5):#range(df.shape[0]):
        # list of a few cats pertaining to this restaurant
        rest_categories = df.iloc[i]['categories']

        for cat in rest_categories:
            if cat in top_categories:
               inv_idx[cat].append(i)
              
    return inv_idx

       
def create_top_category_vectors(
    input_review_vectors: np.ndarray,
    inverted_index: Dict,
    top_categories: List[str],
    n_good_types: int) -> np.ndarray:
    """Returns a numpy array of shape n_top_categories by n_good_types such that the 
    entry (ij) indicates the average value of good type j for review vectors that
    pertain to restaurants with categories i

    Parameters
    ----------
    input_review_vectors: np.ndarray
        numpy array of shape n_reviews by n_good_types such that the 
        entry (ij) indicates if review i contains word j (binary). 

    inverted_index: Dict
        key = categories, value = single-linked list of review_ids

    top_categories: List[str]
        list of top categories

    n_good_types: int
        number of good types represented by the vector

    Returns
    -------
    categories_vectors: np.ndarray
        A numpy array of shape n_top_categories by n_good_types
    """    

    #initialize empty array
    category_vectors = np.zeros((len(top_categories), n_good_types))
    for cat in inverted_index.keys():
        i = top_categories.index(cat)
        for review_id in inverted_index[cat]:
          category_vectors[i] += input_review_vectors[review_id]
        # divide by number of reviews with that cat to take the avg
        # add one to avoid dividing by 0
        category_vectors[i] /= (len(inverted_index[cat]) + 1)
    return category_vectors

def create_query_vector(top_categories: List[str],
    input_category_vectors: np.ndarray,
    input_selected_categories: List[str],
    n_good_types: int) -> np.ndarray:
    """Returns a numpy array vector of length n_good_types such that the 
    ith entry indicates the weight of that type in the query

    Parameters
    ----------
    top_categories: List[str]
        list of top categories

    input_category_vectors: np.ndarray
        numpy array of shape n_categories by n_good_types such that the 
        entry (ij) indicates the weight of word j in restaurant type i

    input_selected_categories: List[str]
       list of categories selected by the user

    n_good_types: int
        number of good types represented by the vector

    Returns
    -------
    query_vector: np.ndarray
        A numpy array of shape n_top_categories by n_good_types
    """
    # initialize empty array
    query_vector = np.zeros(n_good_types)
    # for each category selected by the user, add that category's vector to the query vector
    #(all are weighted equally)
    for cat in input_selected_categories:
       query_vector += input_category_vectors[top_categories.index(cat)]
    return query_vector
       

    


