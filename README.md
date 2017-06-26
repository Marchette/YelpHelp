# YelpHelp
YelpHelp is a datascience project completed during my time as an Insight Data Science Fellow. The goal is to find Yelp "tipping point reviews that predict a sudden decline in ratings. 

The Jupyter notebook "Find Tipping Points" uses probabilistic programming to discover restaurants that have distinct switchpoints where there's a change in average rating. I then use these "tipping point" labels to build a logistic regression to classify whether an individual review comes from a stable period or from a tipping point in the "YelpHelp" Jupyter Notebook.

Note, to run these notebooks you must first create Postgres SQL databases called "businesses" and "reviews" and populate them with the contents of the files: "businesses.sql" and "reviews.sql."
