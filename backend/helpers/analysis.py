from typing import List, Tuple, Dict
from collections.abc import Callable
import numpy as np
import pandas as pd
import re

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
        A numpy array of shape n_speakers by n_good_types such that the 
        entry (ij) indicates how often speaker i says word j.
    """
    word_occurence_matrix = np.zeros((input_df.shape[0], len(input_good_types)))
    good_types = np.array(input_good_types)

    for i in range(input_df.shape[0]):
       tokens = tokenize_method(input_df.iloc[i]['text'])
       for token in tokens:
          if token in input_good_types:
             word_occurence_matrix[i][input_good_types.index(token)] = 1
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
    print("empty dict", inv_idx)
    for i in range(review_vectors.shape[0]):
       review_index = i
       review_id = input_df.iloc[i]["review_id"]
       for j in range(len(input_good_types)):
          word = input_good_types[j]
          freq = review_vectors[i][j]
          
          if freq != 0:
            inv_idx[word].append((review_index, review_id, freq))
    return inv_idx


          
       
       


       
       
       

    


