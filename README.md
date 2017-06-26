# YelpHelp
YelpHelp is a data science project completed during my time as an Insight Data Science Fellow. The goal is to find "tipping point" reviews that predict a sudden decline in ratings. 

The Jupyter notebook "Find Tipping Points" uses probabilistic programming to discover restaurants that have distinct switchpoints where there's a change in average rating. I then use these "tipping point" labels to build a logistic regression to classify whether an individual review comes from a stable period or from a tipping point in the "YelpHelp" Jupyter Notebook.

The model trained in the YelpHelp notebook is saved in the three files: vectorizer.pk, logistic_weights, and scale_stats.pk. These three pickled files allow me to load the pre-trained model on the website and then run evaluate it on another set of reviews. 

The yelphelp folder contains the package of data processing, analysis, and visualization functions I wrote for this project. Both of the notebooks and the webapp require modules from the package. 

The tipping folder (for tipping point) contains the Flask code for the yelphelp.guru site. The main file within the folder is views.py which defines the main functions of the website. tipping/templates contains the html for the different pages, and the tipping/static folder contains the CSS stylesheets and images.

Note, to run these notebooks you must first create Postgres SQL databases called "businesses" and "reviews" and populate them with the contents of the files from my local machine: "businesses.sql" and "reviews.sql." These are too large to host here. 
