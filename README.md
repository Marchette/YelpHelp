# YelpHelp
YelpHelp is a data science project completed during my time as an Insight Data Science Fellow. The goal is to find "tipping point" reviews that predict a sudden decline in ratings. 

The Jupyter notebook "Find Tipping Points" uses probabilistic programming to discover restaurants that have distinct switchpoints where there's a change in average rating. I then use these "tipping point" labels to build a logistic regression to classify whether an individual review comes from a stable period or from a tipping point in the "YelpHelp" Jupyter Notebook.

The yelphelp folder folder contains the package of data processing, analysis, and visualization functions I wrote for this project. Both of the notebooks and the webapp require modules from the package. 

Note, to run these notebooks you must first create Postgres SQL databases called "businesses" and "reviews" and populate them with the contents of the files from my local machine: "businesses.sql" and "reviews.sql." These are too large to host here. 
