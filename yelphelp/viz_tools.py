"""This module is for visualizing the YelpHelp reviews. It contains the functions: make_roc, importance_plot,
rolling_average_rating, and comparison_wordcloud"""

from wordcloud import WordCloud
from collections import Counter
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate



from bokeh.plotting import figure, Figure, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.tickers import FixedTicker

def make_roc(answers, predictions, plot_me=0):
    """Creates an ROC curve. First input should be the true labels, and the second should be the prediction
    probabilities output from a model. A third option plot_me can be specified: if equal to one then this
    function will also output an ROC curve figure"""
    false_positives, true_positives, thresholds = roc_curve(answers, predictions, sample_weight=None)
    roc_auc = roc_auc_score(answers, predictions, average='weighted')
    if plot_me == 1:
        plt.figure(frameon=False)
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = False

        plt.plot(false_positives, true_positives, color='darkorange', lw=2, label='ROC (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=16)

        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)

        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

        plt.axes().set_aspect('equal')
        plt.show()


def importance_plot(dates, ratings, importance, reviews, bad_words):
    """This plot generates the figure seen on the YelpHelp website. It plots a review history, in which potential
    tipping point reviews are plotted with inflated circles. It also defines a hover-tool tip allowing one to
    visualize the full text of the review by mousing over a point, as well as the keywords the model found important
    contained within the review."""

    min_size = 8
    max_growth = 32

    importance = np.round((max_growth * importance) + min_size)

    source = ColumnDataSource(
        data=dict(
            desc=reviews,
            bad=bad_words,
            ratings_source=np.zeros(len(ratings))
        )
    )

    # Create the figure!
    title_string = 'Review History'
    review_plot = Figure(plot_width=600, plot_height=600, x_axis_type="datetime", tools=[], responsive=True,
                         toolbar_location=None,webgl=True,title=title_string)

    cr = review_plot.circle(dates, ratings, size=importance,
                            color="navy", hover_fill_color="firebrick",
                            fill_alpha=0.9, hover_alpha=0.9,
                            line_color=None, hover_line_color="white", source=source)

    hover = HoverTool(

        tooltips="""
        <link href="../static/css/bootstrap.min.css" rel="stylesheet">
        <div>

            <div style="font-family: verdana; width : 550px; position: fixed; left: 650px; top: 180px; padding: 10px">

                <span style="font-size: 17px;"> <b>Review: </b> @desc</span>
            </div>
            <div style="font-family: verdana; width : 550px; position: fixed; left: 650px; top: 120px; padding: 10px">
                <span style="font-size: 14px; font-weight: bold;"> Keywords: @bad</span>
            </div>
        </div>""",
        renderers=[cr]
    )

    review_plot.add_tools(hover)

    newdate = pd.date_range(dates.max(), periods=5, freq='120D')

    projected_mean = 0  # estimated_switch[-1]
    projection_uncertainty = 20  # np.round(estimated_sigma * 2 * 10)

    review_plot.xaxis.axis_label = "Date"
    review_plot.xaxis.axis_line_width = 3
    review_plot.xaxis.axis_label_text_font_style = "bold"
    review_plot.xaxis.axis_label_text_font_size = '20pt'
    review_plot.xaxis.major_label_text_font_size = '16pt'

    review_plot.yaxis.axis_line_width = 3
    review_plot.yaxis.axis_label_text_font_style = "bold"
    review_plot.yaxis.axis_label_text_font_size = '20pt'
    review_plot.yaxis.ticker = FixedTicker(ticks=[-3, -2, -1, 0, 1, 2, 3])
    review_plot.yaxis.major_label_text_font_size = '16pt'
    review_plot.xgrid.visible = False
    review_plot.ygrid.visible = False
    review_plot.outline_line_alpha = 0

    review_plot.xaxis.axis_label = 'Date Posted'
    review_plot.yaxis.axis_label = 'Rating (Normed to User Average)'

    review_plot.title.text_font_size = '16pt'
    review_plot.title.align='center'

    #rolling_ratings, edited_dates, interp_ratings, samplerange = rolling_average_rating(business_reviews, 100)

    # show the results
    return review_plot


def rolling_average_rating(reviews, sample_resl):
    """This function calculates a rolling window average on a list of reviews. You must input the reviews as
    a Pandas dataframe with both dates and the ratings given, and also input a sampling resolution"""
    # N: Minimum number of ratings to consider
    # sample_resl: how many bins to look at
    sample_range = np.arange(0, 1, 1 / sample_resl)
    ratings = []
    interpolated_ratings = []
    dates = []

    ratings = reviews['user_normed_stars']
    dates = reviews['date']
    dates = dates - dates.min()
    dates = dates.astype(datetime.datetime)
    dates = dates / dates.max()

    # These are already presorted
    # date_order = np.argsort(dates)
    # dates = dates.iloc[date_order]
    # ratings = ratings.iloc[date_order]

    ratings = ratings.rolling(window=30, center=True, win_type='gaussian', min_periods=1).mean(std=8)

    interpolate_ratings = interpolate.interp1d(dates, ratings)
    interpolated_ratings = pd.Series(interpolate_ratings(sample_range))

    return ratings, dates, interpolated_ratings, sample_range


def comparison_wordcloud(text1 ,text2):
    """This function takes two text documents and then makes a wordcloud based on their differential frequency;
    that is, larger words are more common in document 1 than document 2"""
    # Returns a dictionary of words that are more frequent in text1 than text2
    wc_color_map = 'plasma'

    all_text1 = [word for doc in text1 for word in doc.split()]
    all_text2 = [word for doc in text2 for word in doc.split()]

    all_words = []
    all_words.extend(all_text1)
    all_words.extend(all_text2)

    all_word_counter = dict(Counter(all_words))
    text1_counter = dict(Counter(all_text1))
    text2_counter = dict(Counter(all_text2))

    differential_counter = {}
    for key in all_word_counter.keys():
        if key not in text1_counter:
            text1_counter[key] = 0
        if key not in text2_counter:
            text2_counter[key] = 0

        freq_difference = text1_counter[key ] -text2_counter[key]

        if freq_difference < 0:
            freq_difference = 0
        differential_counter[key] = freq_difference

    wordcloud = WordCloud(stopwords=stopword_set ,colormap = wc_color_map, background_color='white').generate_from_frequencies(differential_counter)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

