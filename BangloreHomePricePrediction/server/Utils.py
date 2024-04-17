import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)
def load_saved_artifacts():
    global __locations
    global __data_columns
    global __model
    print("loading artifacts")
    with open('./artifacts/columns.json', 'r', encoding='utf-8') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:  # Open in binary mode for pickle
        __model = pickle.load(f)


if __name__ == "__main__":
    load_saved_artifacts()
    print("get_location_names",get_location_names())