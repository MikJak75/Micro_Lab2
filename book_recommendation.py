#!/usr/bin/env python

import pandas as pd
import numpy
import pprint
import regex as re
import fasttext.util
import sys
import string
import nltk
from nltk.tokenize import word_tokenize
import fasttext
import wakepy
import numpy as np
import pprint
from numpy import dot
from numpy.linalg import norm




def norm_token(text):
    # normalize string
    text = text.lower()
    text = re.sub(r'(\s+)(a|an|the)(\s+)', r'\1\3', text)
    #text = text.translate(str.maketrans('','',string.punctuation))
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    #text = re.sub(r'(\s+)(.|n|the)(\s+)', r'\1\3', text)

    #tokenize into wordes 
    word_list = word_tokenize(text)

    return word_list


def process_books(filename):
    full_books = pd.read_csv(filename)

    #filter by genre
    genre_filter = [('Mystery' in x and 'Thriller' in x) for x in full_books['genres'].values ]
    books = full_books[genre_filter]

    books = books[['title', 'description', 'genres']]
    books = books.dropna()

    #make new dataframe which has one column that is a combination of title and description
    combined_books = books['genres'] + ' ' + books['description']

    combined_books = list(combined_books)

    combined_books_nt = []

    for x in range(len(combined_books)):
        combined_books_nt.append(norm_token(combined_books[x]))
        
    #books is a dataframe and combined_books is a series
    return books, combined_books_nt

# take in a list of words and return an average wordvec of all the words
def average_wordvec(words, ft_model):
    wv_array = np.array([ft_model[x] for x in words])

    return np.mean(wv_array, axis=0)

def avg_vectorize_books(combined_books_nt, ft_model):
    combined_books_avgvecs = []
    for token_list in combined_books_nt:
        avg_wv = average_wordvec(token_list, ft_model)
        combined_books_avgvecs.append(avg_wv)

    
    return np.array(combined_books_avgvecs)


def cos_sim(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim


def compare_cos_sim(target, wordvecs):
    similarities = np.array([cos_sim(target, wv) for wv in wordvecs])

    return similarities


def main():
    if (len(sys.argv) != 2):
        print("Usage Error: Must have 1 string argument")
        exit()

    target_description = sys.argv[1]
    target_description_nt = norm_token(target_description)

    books, combined_books_nt = process_books('books_1.Best_Books_Ever.csv')

    ft_model = fasttext.load_model('cc.en.100.bin')

    target_description_avgvec = average_wordvec(target_description_nt, ft_model)

    combined_books_avgvecs = avg_vectorize_books(combined_books_nt, ft_model)

    similarities = compare_cos_sim(target_description_avgvec, combined_books_avgvecs)

    most_similar_index = np.argmax(similarities)

    similarity_score = similarities[most_similar_index]
    title = books.iloc[most_similar_index]['title']


    print(f'Similarity Score: {similarity_score}')
    print(title)


if __name__ == '__main__':
    main()