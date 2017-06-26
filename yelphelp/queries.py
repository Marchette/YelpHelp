"""Functions for querying SQL databases containing reviews and business information:
query_business_reviews, query_business_switchpoints"""

import pandas as pd

#Get all of the reviews for a business from the SQL database
def query_business_reviews(business_id,connection):
    """Get all of the reviews for a specific business"""
    sql_query = """
    SELECT business_id,user_normed_stars, stars,date, useful, text FROM reviews_data_table WHERE business_id = '""" + business_id + """';
    """
    reviews_data_from_sql = pd.read_sql_query(sql_query,connection)

    #Sort them so we can accurately look for switchpoints
    reviews_data_from_sql['date'] = pd.DataFrame(pd.to_datetime(reviews_data_from_sql.date))
    reviews_data_from_sql = reviews_data_from_sql.sort_values(by='date')

    return reviews_data_from_sql

def query_business_switchpoints(connection):
    """Get the switchpoint associated with each business"""
    sql_query = """
    SELECT * FROM switchpoint_data_table;
    """
    switches_from_sql = pd.read_sql_query(sql_query,connection)
    return switches_from_sql
