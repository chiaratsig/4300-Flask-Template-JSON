from typing import List, Tuple, Dict
from collections.abc import Callable
import numpy as np
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
    """Returns the set of of distinct tokens used in an entire transcript

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
      tokens += tokenize(input_reviews["text"][review_idx])
    return set(tokens)

def num_dedup_tokens(tokenize_method: Callable[[str], List[str]],
    tokenize_transcript_method: Callable[[Callable[[str], List[str]], Tuple[str, List[Dict[str, str]]]], List[str]],
    input_transcripts: List[Tuple[str, List[Dict[str, str]]]]) -> int:
    """Returns number of tokens used in an entire transcript

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
    int
        The total number of tokens in all the transcripts.
    """
    # TODO-2.3
    tokens = []
    for transcript in input_transcripts:
      tokens += tokenize_transcript_method(tokenize_method, transcript)
    return len(tokens)

def build_word_count(tokenize_method: Callable[[str], List[str]],
    tokenize_transcript_method: Callable[[Callable[[str], List[str]], Tuple[str, List[Dict[str, str]]]], List[str]],
    input_transcripts: List[Tuple[str, List[Dict[str, str]]]], 
    input_titles: Dict[str, str]) -> Dict[str, int]:
    """Returns a dictionary with the number of episodes each distinct word appears

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
    input_titles: Dict[str, str]
        A dictionary of transcript UIDs mapped to corresponding episode titles.

    Returns
    -------
    Dict[str, int]
        A dictionary of words mapped to the number of episodes they appear in.
    """
    title_tokens = {}
    for transcript in input_transcripts:
      title = input_titles[transcript[0]]
      if title not in title_tokens.keys():
        title_tokens[title] = set()
      tokens = tokenize_transcript_method(tokenize_method, transcript)
      for token in tokens:
        title_tokens[title].add(token)
    
    word_episode_counts = {}
    for title in title_tokens.keys():
      for word in title_tokens[title]:
        if word not in word_episode_counts.keys():
          word_episode_counts[word] = 0
        word_episode_counts[word] += 1
    
    return word_episode_counts

def build_word_episode_distribution(input_word_counts: Dict[str, str]) -> Dict[int, int]:
    """Returns a dictionary that counts how many words appear in exactly a given number of episodes

    Parameters
    ----------
    input_word_counts : Dict[str, str]
        A dictionary of word mappeds to the number of episodes they appear in.

    Returns
    -------
    Dict[int, int]
        A dictionary that maps a number of episodes to the number of words that appear in that many episodes.
    """
    dist = dict()
    for key in input_word_counts.keys():
      count = input_word_counts[key]
      if count not in dist.keys():
        dist[count] = 0
      dist[count] += 1
    return dist

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

def create_ranked_good_types(tokenize_method: Callable[[str], List[str]],
    tokenize_transcript_method: Callable[[Callable[[str], List[str]], Tuple[str, List[Dict[str, str]]]], List[str]],
    input_transcripts: List[Tuple[str, List[Dict[str, str]]]], 
    input_good_types: List[str]) -> List[Tuple[str, float]]:
    """Returns a list of good types in reverse sorted order in the form:
        [(word_1,word_frequency_1),
        ...
        (word_10,word_frequency_10)]

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
    input_good_types: List[str]
        A list of all good types, sorted alphabetically.

    Returns
    -------
    List[Tuple[str, float]]
        A list of tuples containing a good type and its frequency, sorted in descending order.
    """
    def sort_tuple(tup):
      """ Returns a list of tuples sorted in descending order by the second element in the tuple."""
      for x in range(0, len(tup)):
          for y in range(0, len(tup)-x-1):
              if (tup[y][1] < tup[y + 1][1]):
                  temp = tup[y]
                  tup[y] = tup[y + 1]
                  tup[y + 1] = temp
      return tup
    num_tokens = 0
    word_frequency = dict()
    for transcript in input_transcripts:
      tokens = tokenize_transcript_method(tokenize_method, transcript)
      num_tokens += len(tokens)
      for token in tokens:
        if token not in word_frequency.keys():
          word_frequency[token] = 0
        word_frequency[token] += 1

    good_word_frequencies = []
    for word in input_good_types:
      freq = round(word_frequency[word]/num_tokens, 5)
      good_word_frequencies.append((word, freq))
    return sort_tuple(good_word_frequencies)
    

def create_word_occurrence_matrix(
    tokenize_method: Callable[[str], List[str]],
    input_transcripts: List[Tuple[str, List[Dict[str, str]]]],
    input_speakers: List[str],
    input_good_types: List[str]) -> np.ndarray:
    """Returns a numpy array of shape n_speakers by n_good_types such that the 
    entry (ij) indicates how often speaker i says word j.

    Parameters
    ----------
    tokenize_method : Callable[[str], List[str]]
        A method to tokenize a string into a list of strings representing words.
    input_transcripts : List[Tuple[str, List[Dict[str, str]]]]
        A list of tuples containing a transcript UID mapped to a list of utterances 
        (specified as a dictionary with ``speaker``, ``text``, and ``timestamp`` as keys).
    input_speakers: List[str]
        A list of speaker names
    input_good_types: List[str]
        A list of all good types, sorted alphabetically.

    Returns
    -------
    np.ndarray
        A numpy array of shape n_speakers by n_good_types such that the 
        entry (ij) indicates how often speaker i says word j.
    """
    word_occurence_matrix = np.zeros((len(input_speakers), len(input_good_types)))
    speakers = np.array(input_speakers)
    good_types = np.array(input_good_types)

    for transcript in input_transcripts:
      for utterance in transcript[1]:
        speaker = utterance["speaker"]
        text = utterance["text"]
        speaker_idx = np.where(speakers == speaker)
        assert len(speaker_idx[0]) == 0 or 1
        if len(speaker_idx[0] == 1):
          tokens = tokenize_method(text)
          for token in tokens:
            word_idx = np.where(good_types == token)
            assert len(word_idx[0]) == 0 or 1
            if len(word_idx[0]) == 1:
              word_occurence_matrix[speaker_idx[0][0]][word_idx[0][0]] += 1
      
    return word_occurence_matrix

def create_weighted_word_freq_array(input_word_array: np.ndarray) -> np.ndarray:
    """Returns a numpy array of shape n_speakers by n_good_types such that the 
    entry (ij) indicates how often speaker i says word j weighted by the above ratio.
    
    Note: You must add 1 to the sum of each column to avoid divison by 0 issues.

    Parameters
    ----------
    input_word_array: np.ndarray
        A numpy array of shape n_speakers by n_good_types such that the 
        entry (ij) indicates how often speaker i says word j.

    Returns
    -------
    np.ndarray
        A numpy array of shape n_speakers by n_good_types such that the 
        entry (ij) indicates how often speaker i says word j
    """
    # TODO-3.3
    return input_word_array / (input_word_array.sum(0) + 1)


# different from the one in a4
def build_inverted_index(df: List[dict]) -> dict:
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
    for index, row in df.iterrows():
        if row['business_id'] in inv_idx.keys():
          inv_idx[row['business_id']].append((index, row['review_id']))
        else:
          inv_idx[row['business_id']] = list((index, row['review_id']))


       
       
       

    


