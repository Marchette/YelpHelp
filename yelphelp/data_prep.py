"""Functions to prepare data for Yelp Help:
get_business_class, get_features, get_ratings_history,
sample_reviews, scale_features"""

import numpy as np
from .queries import query_business_switchpoints, query_business_reviews
from .nlp_tools import nlp_preprocess_review, get_local_sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

def get_business_class(differential, switch_threshold=.75):
    """Assigns labels to a business: if the differential is above the switch threshold
    (default: .75) then it's a tipping point, if it's under the negative threshold its
    an increase, and if it is neither it is stable"""
    if differential > switch_threshold:
        classification = 1
    elif differential < -switch_threshold:
        classification = -1
    else:
        classification = 0
    return classification

def get_features(review, index):
    """This function calculates a number of statistics about a review, including the rating given,
    the user-normed rating, how many people found the review useful, the number of words, as well
    as statistics about the rating relative to the recent history of the restaurant"""
    stars = np.array(review['stars'][index])
    normed_stars = np.array(review['user_normed_stars'][index])
    useful = np.array(review['useful'][index])
    review['num_words'] = review['text'].apply(lambda x: len(x.split()))
    num_words = np.array(review['num_words'][index])

    current_mean_rating, mean_rating_slope, std_rating, rating_ratio = get_rating_history(
        review['user_normed_stars'][np.arange(0, index)])

    features = np.array([stars, normed_stars, useful, num_words])

    features = np.hstack([features, current_mean_rating, mean_rating_slope, std_rating, rating_ratio])
    return features

def get_rating_history(ratings, N=10):
    """Calculates the mean rating of the past N reviews from the end of the input (not including the last review)"""
    ratings_history = ratings.iloc[:-1]

    if len(ratings_history) < 5:
        ratings_smoothed = ratings_history.rolling(window=30, center=True, win_type='gaussian', min_periods=1).mean(std=8)

        # We want to get the current mean rating
        current_mean_rating = ratings_smoothed.iloc[-1]

        # We also want to get the slope of the mean ratings
        if N > len(ratings_smoothed):
            N = len(ratings_smoothed)

        mean_rating_slope = (ratings_smoothed.iloc[-N] - ratings_smoothed.iloc[-1]) / N

        std_rating = np.std(ratings_history.iloc[-N:])

        rating_ratio = (1 + ratings.iloc[-1]) / (1 + current_mean_rating)
    else:
        current_mean_rating = 0
        mean_rating_slope = 0
        std_rating = 0
        rating_ratio = 0

    return current_mean_rating, mean_rating_slope, std_rating, rating_ratio


def sample_reviews(business_df, switchpoint, N):
    """This function grabs all of the reviews within N positions of a switchpoint, allowing us to build
    a dataset containing reviews at tipping points as well as reviews at stable periods. It also creates
    some useful features regarding the reviews it includes and preprocesses their text. It returns the
    processed text and these features."""

    # Switchpoint: is an integer saying where the max a post. estimate of the switch is
    # N: is how many reviews to grab at this location

    # num_context: this is used in the server sentiment analysis to define how many words around
    # the mention of the server we want to gather for the sentiment analysis
    num_context = 5
    analyzer = SentimentIntensityAnalyzer()
    reviews = business_df['text']
    ratings = business_df['user_normed_stars']

    # Make a list of all the N closest reviews to switchpoint
    review_indicies = np.ceil(np.arange(switchpoint - (N / 2), 1 + switchpoint + (N / 2)))

    review_indicies = list(filter(lambda x: np.logical_and(x < len(reviews), x >= 0), review_indicies))

    # And we grab all the reviews we want to
    review_samples = []
    review_features = np.empty((14))

    if len(review_indicies) > 0:
        for review_index in review_indicies:
            features = get_features(business_df, review_index)
            processed_review = nlp_preprocess_review(reviews[review_index])

            service_sentiment = get_local_sentiment(processed_review, 'service', search_context=5)
            #            food_sentiment get_local_sentiment(processed_review,'food',search_context=5):

            # Also grab the 5 previous reviews and append them together

            sentiment = list(analyzer.polarity_scores(reviews[review_index]).values())
            sentiment = np.array(sentiment)
            sentiment_ratio = (1 + sentiment[2]) / (1 + sentiment[0])  # ratio of positive to negative
            sentiment = np.append(sentiment, sentiment_ratio)
            sentiment = np.append(sentiment, service_sentiment)

            # Add sentiment to features
            features = np.append(sentiment, features)

            review_samples.append(processed_review)
            review_features = np.vstack([review_features, features])

        review_features = review_features[1:]

    return review_samples, review_features

def scale_features(features,is_training=True):
    """This function normalizes all numeric features to be mean 0 and have a similar range"""
    if is_training==True:
        scaling_mean = np.mean(features,axis=0)
        scaled_features = features-scaling_mean
        scaling_range = np.max(scaled_features,axis=0)-np.min(scaled_features,axis=0)
        scaled_features = scaled_features / (1+scaling_range)
        with open('scale_stats.pk', 'wb') as fin:
            pickle.dump( (scaling_mean,scaling_range) , fin)
    elif is_training==False:
        scale_stats = pickle.load(open( "scale_stats.pk", "rb" ) )
        scaling_mean = scale_stats[0]
        scaling_range = scale_stats[1]
        scaled_features = (features-scaling_mean) / (1+scaling_range)
    return scaled_features