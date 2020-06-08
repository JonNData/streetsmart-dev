# Run app by typing "uvicorn main:app --reload" in terminal
# --reload lets you hot reload your app after making changes

from fastapi import FastAPI
import psycopg2
import os
import pickle
from dotenv import load_dotenv
import pandas as pd
from rapidfuzz import fuzz, process
from joblib import load
import requests
import numpy as np


load_dotenv()

#Load credentials from .env
name = os.environ["DB_NAME_AWS"]
password = os.environ["DB_PW_AWS"]
host = os.environ["DB_HOST_AWS"]
user = os.environ["DB_USER_AWS"]

pg_conn = psycopg2.connect(dbname=name,
                        user=user,
                        password=password,
                        host=host
                        )
## Curson is always open
pg_curs = pg_conn.cursor()

app = FastAPI()

# Load in slimmed random forest pickled model
#test_model = pickle.load(open("random_forest_1.sav" , "rb"))
test_model = load("targetiterrobustforest.joblib")

# Load the craigslist cleaned data
df_cl = pd.read_csv("https://raw.githubusercontent.com/Lambda-School-Labs/street-smarts-ds/master/data/model_and_image_url_lookup.csv")
# List of unique CL cars
cl_models = sorted(df_cl.model.unique())

@app.get("/")
def root():
    return {"Street": "Smarts",
            "Version": "Developer",
            "API": "/docs"}


@app.post('/predict')
def predict(make:str="Ford", model:str="F150 Pickup 4WD", year:int=2005):
    """
    Predict price based on year, manufacturer, model
    """
    manufacturer = make
    manufacturer = manufacturer.lower()
    model_lower = model.lower()

    # use fuzzy wuzzy to get the closest match
    model_fz = process.extractOne(model_lower, cl_models, scorer=fuzz.token_sort_ratio)[0]
    
    input = pd.DataFrame({
        "year": [year],
        "manufacturer": [manufacturer],
        "model": [model_fz],
        "odometer": 10000
        })
    
    pred = test_model.predict(input)

    car_price_prediction = pred[0]
    
    '''
     still need:

     federal state local incentives
     state sales tax
     '''

    #### 5 Year Cost to Own, with CO2
    ## Constant values
    ## These can be user inputs in a later release canvas

    miles_per_year = 15000
    num_years = 5
    # mpg = 25
    gas_cost = 3
    electric_cost = .12
    maintenance_cost_per_year = 1000

    # Query the database for combined mpg
    pg_curs.execute(f"select AVG(comb08) FROM epa_vehicles_all WHERE make = '{make}' and model = '{model}' and year = '{year}';")
    mpg_combined = pg_curs.fetchall()[0][0]

    # Query the database for CO2
    pg_curs.execute(f"SELECT AVG(co2tailpipegpm) FROM epa_vehicles_all WHERE make = '{make}' AND model = '{model}' AND year = {year};")
    co2_grams_per_mile = pg_curs.fetchall()[0][0]

    ## CO2 over a X year period (Kg)
    co2_over_time = co2_grams_per_mile * miles_per_year  * num_years / 1000

    ## Fuel, maintenance, and 5 yr cost
    fuel_cost = (miles_per_year / mpg_combined * gas_cost * num_years)
    maintenance_cost = maintenance_cost_per_year * num_years

    five_year_cost_to_own = (
    car_price_prediction + fuel_cost + maintenance_cost
    )
    # Can eventually query EPA db and use fuel_type to determine fuel_cost and maintenance_cost for EVs

    ## Number of kgs of CO2 absorbed by one tree per year
    tree_absorption = 21.7724
    number_of_trees_to_offset = co2_over_time/(tree_absorption*num_years)

    #### Images of Selected Car

    def status_200_or_nan(url):
        response = requests.get(url)
        if response.status_code == 200:
            return url
        else:
            return np.NaN

    def year_to_urls(car_model, year):
        """
        input cl car and year, output a list of working urls
        """
        df_models = df_cl[df_cl['model'] == car_model]
        df_models_at_year = df_models[df_models['year'] == year]
        index_of_model_year = df_models_at_year.index[0:4]

        list_urls = list(df_cl['image_url'][index_of_model_year])
        list_w_nan = [status_200_or_nan(x) for x in list_urls]
        clean_list_urls = [x for x in list_w_nan if x is not np.NaN]
        return clean_list_urls

    def fetch_image(car_model, year):
        """
        Check other years images if needed
        """
        clean_list_urls = year_to_urls(car_model, year)
        #if list empty, check other years
        if len(clean_list_urls) == 0:
            year1 = year + 1
            clean_list_urls = year_to_urls(car_model, year1)

            if len(clean_list_urls) == 0:
                year0 = year - 1
                clean_list_urls = year_to_urls(car_model, year0)

                # no car image
                if len(clean_list_urls) == 0:
                    return ['https://raw.githubusercontent.com/Lambda-School-Labs/street-smarts-ds/master/data/noImage_large.png']
                return clean_list_urls
            return clean_list_urls
        return clean_list_urls


    list_of_imgs = fetch_image(model_fz, year)

    return {"car_price_prediction": car_price_prediction.round(2),
            "fuel_cost": round(fuel_cost, 2),
            "maintenance_cost": maintenance_cost,
            "five_year_cost_to_own": five_year_cost_to_own.round(2),
            "co2_five_year_kgs": round(co2_over_time, 2), 
            "number_of_trees_to_offset": round(number_of_trees_to_offset, 0),
            "list_of_imgs": list_of_imgs}

@app.post("/carbon_emissions2")
def get_co2_sql(make:str="Chevrolet", model:str="Sonic", year:int=2018):

    """
    Return the co2 value for the inputted vehicle base on make, model, and year
    """

    pg_curs.execute(f"SELECT AVG(co2tailpipegpm) FROM epa_vehicles_all WHERE make = '{make}' AND model = '{model}' AND year = {year};")
    value = pg_curs.fetchall()[0][0]
    #pg_curs.close()

    return {"predicted_co2_sql": value}
