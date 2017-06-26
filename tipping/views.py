import sys
import time
sys.path.append('/home/ubuntu/application/tipping')

from .yelphelp.viz_tools import importance_plot
from .yelphelp.nlp_tools import nlp_preprocess_review, create_nlp_features, get_pos_dist, service_complaint, get_local_sentiment
from .yelphelp.data_prep import get_features, scale_features
from .yelphelp.queries import query_business_reviews

import flask
from flask import Flask, render_template, request
from tipping import app

import psycopg2

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy import sparse
from scipy import interpolate
import scipy as sp
import scipy.io
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.embed import components
from bokeh.models import HoverTool
from bokeh.models.tickers import FixedTicker

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn import model_selection, preprocessing, ensemble

from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pymc3 as pm
from pymc3 import Model, DiscreteUniform, Exponential, Poisson, Normal, HalfNormal, NUTS, Metropolis, sample, traceplot, find_MAP
from pymc3.math import switch

import pickle

reviews_database = 'reviews1000'
business_database = 'businesses'
username = 'postgres' ##This must matches PostgreSQL' user name

#Connect to the business and reviews databases
business_con = None
business_con = psycopg2.connect(database = business_database, user = username)
review_con = None
review_con = psycopg2.connect(database = reviews_database, user = username)

#Start by loading all of the business_ids
sql_query = """
SELECT * FROM business_data_table WHERE categories LIKE '%Restaurants%' AND longitude < -60 AND review_count > 40;
"""

business_data_from_sql = pd.read_sql_query(sql_query,business_con)

#Create the index page
@app.route('/')
@app.route('/index')
def main():
    return render_template("input.html")

#Create the page displaying the review history of a business
@app.route('/output')
def run_plot():

    #If the user left the business id form blank return them to index page
    businum = request.args.get('busi_id')
    if not businum:
        return render_template('input.html')

    #Remove any non-digits from the business id
    cleannum = []
    [cleannum.append(letter) for letter in businum if letter.isdigit()]
    businum = ''.join(cleannum)
    businum = int(businum)

    businum = np.mod(businum,1000)

    #Get all of the reviews for this business
    start_time = time.time()
    business_reviews = query_business_reviews(business_data_from_sql['business_id'][businum],review_con)
    num_reviews = len(business_reviews['stars'])

    query_time = time.time()
    print('Query time: ' + str(query_time-start_time))

    #Go through and concatenate all the reviews together (doing our normal preprocessing)
    review_samples = []
    review_fulltext = []
    review_features = np.empty( (14,) )
    analyzer = SentimentIntensityAnalyzer()

    start_time = time.time()
    for review_num in np.arange(0,num_reviews):

        if review_num > 10:
            features = get_features(business_reviews,review_num)
        else:
            features = [0,0,0,0,0,0,0,0]

        sentiment = analyzer.polarity_scores(business_reviews['text'][review_num])
        sentiment = list([sentiment['neg'],sentiment['neu'],sentiment['pos'],sentiment['compound']])
        sentiment = np.array(sentiment)

        sentiment_ratio = (1+sentiment[2]) / (1+sentiment[0]) #ratio of positive to negative
        sentiment = np.append(sentiment, sentiment_ratio)

        processed_review = nlp_preprocess_review(business_reviews['text'][review_num])
        service_sentiment = get_local_sentiment(processed_review,'service',search_context=5)
        sentiment = np.append(sentiment, service_sentiment)

        #Add sentiment to features
        features = np.append(sentiment, features)

        review_fulltext.append(business_reviews['text'][review_num])
        review_samples.append(processed_review)
        review_features = np.vstack( [review_features, features]) 

    analysis_time = time.time()
    print('Analysis time: ' + str(analysis_time-start_time) )
    review_features = review_features[1:]

    #Apply the Tf-Idf vectorizer we created in the YelpHelp Jupyter notebook
    tfid_reviews, nlp_feature_names = create_nlp_features(review_samples,is_training=False)

    #Open the model we trained in the YelpHelp Jupyter notebook
    with open("/home/ubuntu/application/tipping/logistic_weights",'rb') as fin:
        model = pickle.load(fin)

    #Concatenate NLP and quantitative features together!
    start_time = time.time()
    pos_dist = []

    #Get the part of speech distribution
    for review in review_samples:
        pos_dist.append(get_pos_dist(review))

    #Get whether each review mentioned the server or another of our
    #theorized complaints
    service_mentions = []
    for review in review_samples:
        service_mentions.append(service_complaint(review))

    review_features = np.hstack( [review_features, np.array(pos_dist), np.array(service_mentions)] )
    scaled_features = scale_features(review_features,is_training=False)

#    data = sparse.hstack( [tfid_reviews, scaled_features] ).tocsc()
    data = tfid_reviews.tocsc()

    #Get the predicted probabilities of tipping points, and rescale them so we can use their values
    #to inflate points within the importance plot
    prediction_ps = model.predict_proba(data)
    prediction_ps = prediction_ps[:,1]

    prediction_ps[prediction_ps<.5] = 0
    prediction_ps[prediction_ps<=0] = 0

    #prediction_ps = prediction_ps
    all_feature_names = np.array(nlp_feature_names)

#    feature_names = ['negative', 'neutral', 'positive', 'composite', 'positive to negative ratio', 'service sentiment', 'rating', 'normed_rating', 'useful', 'num_words', 'current mean rating', 'slope', 'rating standard deviation', 'rating ratio', 'nouns', 'adjectives','adverbs','verbs', 'adjectives:nouns', 'adverb:verb', 'verb:noun', 'server_mentions', 'wait_mentions', 'sick_mentions', 'chef_mentions', 'empty_mentions', 'ill_mentions', 'temperature_mentions', 'change_mentions', 'switch_mentions', 'expensive_mentions', 'forgot_mentions','service_mentions','mistake_mentions']

#    all_feature_names = []
#    all_feature_names.extend(nlp_feature_names)
#    all_feature_names.extend(feature_names)
#    all_feature_names = np.array(all_feature_names)
    all_feature_names = all_feature_names.flatten()

    #Get all of the important features that occured within each review
    bad = []

    bad_feature_ind = np.where( np.abs(model.coef_) > 0.15 )[1]
    bad_feature_names = all_feature_names[bad_feature_ind]

    for review in review_samples:
        bad_words = list(filter(lambda x: x in set(review.split()), bad_feature_names))
        bad.append(bad_words)

    feature_time = time.time()
    print('Feature Time: ' + str(feature_time-start_time) )

    #Create the Bokeh importance plot and then pass what we need for the output page to render it
    start_time = time.time()
    figure = importance_plot(business_reviews['date'], business_reviews['user_normed_stars'], prediction_ps, review_fulltext,bad)
    fig_script, fig_div = components(figure)
    plot_time = time.time()
    print('Plot Time: ' + str(plot_time-start_time) )

    return render_template('data.html', fig_script=fig_script, fig_div=fig_div)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
