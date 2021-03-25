import pandas as pd

def index_util(df) :
    df = df.reindex()
    df = df.reset_index(inplace=True)
    return df


def create_dataframe_from_entity(entity) :
    
    dataframe_path = entity["dataframe_path"]
    dataframe = None
    try :
        dataframe = pd.read_csv(dataframe_path)
    except Exception as e :
        print("Exception {} happended while reading the CSV file".format(e))

    return dataframe, entity['variable_types']

