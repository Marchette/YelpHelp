"""This module contains a number of functions for processing Yelp reviews and engineering features from them:
nlp_preprocess_review, create_nlp_features, get_pos, get_pos_dist, get_local_sentiment, service_complaint"""

import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pickle
from bokeh.plotting import figure, output_file, show
from wordcloud import WordCloud


def nlp_preprocess_review(review):
    """This function tokenizes, lemmatizes, and removes stopwords from a review"""
    # Define the tools we'll use
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    stopword_set = set(stopwords.words('english'))
    stopword_set.remove('not')
    stopword_set.remove('no')
    # stopword_set.add('place')
    # stopword_set.add('food')
    # stopword_set.add('restaurant')

    review = review.lower()
    tokens = tokenizer.tokenize(review)
    preproc_tokens = []
    for token in tokens:
        if not token in stopword_set:
            if not token[0].isdigit():
                token = lemmatizer.lemmatize(token)
                preproc_tokens.append(token)

    words = ' '.join(word for word in preproc_tokens)

    return words

# Combine everything together now
def create_nlp_features(list_of_reviews, is_training=True):
    """This function takes a list of reviews and constructs a Tf-Idf vectorizer from them.
    If is_training=False, then it will load the pickled vectorizer from the training set
    and apply it to the testing set documents"""

    if is_training == True:

        tfid_vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))

        tfid_reviews = tfid_vectorizer.fit_transform(list_of_reviews)
        review_feature_names = tfid_vectorizer.get_feature_names()

        with open('vectorizer.pk', 'wb') as fin:
            pickle.dump(tfid_vectorizer, fin)

    elif is_training == False:
        tfid_vectorizer = pickle.load(open("vectorizer.pk", "rb"))
        tfid_reviews = tfid_vectorizer.transform(list_of_reviews)
        review_feature_names = tfid_vectorizer.get_feature_names()

    return tfid_reviews, review_feature_names


# Gets the part of speech
def get_pos(all_reviews):
    """This function uses NLTK to get the parts of speech from all words in the input"""
    all_review_words = []
    for review in all_reviews:
        for word in review.split():
            all_review_words.append(word)

    tagged = nltk.pos_tag(all_review_words, tagset='universal')
    return tagged


def get_pos_dist(review):
    """This function uses NLTK to calculate the frequency of different parts of speech in a review"""
    tagged = nltk.pos_tag(review.split(), tagset='universal')
    word_tag_freqs = nltk.FreqDist(tag for (word, tag) in tagged)

    # Get just NOUN, ADJ, ADV, VERB
    pos_dist = np.zeros(7)
    pos_dist[0] = word_tag_freqs['NOUN']
    pos_dist[1] = word_tag_freqs['ADJ']
    pos_dist[2] = word_tag_freqs['ADV']
    pos_dist[3] = word_tag_freqs['VERB']
    pos_dist[4] = word_tag_freqs['ADJ'] / (word_tag_freqs['NOUN'] + 1)
    pos_dist[5] = word_tag_freqs['ADV'] / (word_tag_freqs['VERB'] + 1)
    pos_dist[6] = word_tag_freqs['VERB'] / (word_tag_freqs['NOUN'] + 1)

    return pos_dist


def get_local_sentiment(review, search_word, search_context=5):
    """This function performs a sentiment analysis on all of the text in a review that is within
     some distance of around a search word. It takes as input a review, a search_word, and an
     integer search_context"""
    word_bag = review.split()

    analyzer = SentimentIntensityAnalyzer()

    if search_word not in word_bag:
        local_sentiment = 0
    else:
        local_sentiment = 1
        # get just the local neighborhood of N words around the search_word
        search_index = word_bag.index(search_word)
        # Make a list of all the N closest words to searchword
        search_indicies = np.round(np.arange(search_index - (search_context), search_index + (search_context) + 1))

        search_indicies = list(filter(lambda x: np.logical_and(x < len(word_bag), x >= 0), search_indicies))

        review_segment = word_bag[int(np.min(search_indicies)):int(np.max(search_indicies))]
        local_context = ' '.join(review_segment)

        sentiment = analyzer.polarity_scores(local_context)
        sentiment = list([sentiment['neg'], sentiment['neu'], sentiment['pos'], sentiment['compound']])

        local_sentiment = (1 + sentiment[2]) / (1 + sentiment[0])  # ratio of negative to positive

    return local_sentiment

def service_complaint(review):
    """This function detects whether certain words hypothesized to appear at tipping points--like server, chef,
    sick, forgot, etc-- ever appear within a review"""
    word_bag = set(review.split())
    review = nlp_preprocess_review(review)

    check_words = ['server', 'wait', 'sick', 'chef', 'empty', 'ill', 'cold', 'change', 'switch', 'expensive', 'forgot',
                   'service', 'mistake']
    complaint = np.zeros(len(check_words))
    word_count = 0

    for word in check_words:
        these_counts = dict(Counter(review.split()))
        if word in word_bag:
            complaint[word_count] = these_counts[word]

        word_count += 1

    return complaint