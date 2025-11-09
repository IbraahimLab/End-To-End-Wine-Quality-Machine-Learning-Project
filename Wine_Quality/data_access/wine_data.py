from wine_quality.exception import custom_Exception
from wine_quality.constants import  DATABASE_NAME
from wine_quality.configuration.mongo_db_connection import MongoDBClient
import sys
import pandas as pd 

from typing import Optional
import numpy as np




class winedata:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        """
         this place only happens the connection establishment
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise custom_Exception(e,sys)
        

    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            return pd.DataFrame of collection
            """

            """
            this place happpens the fetching and changing into dataframe
            """
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" and "id" in df.columns.to_list():
                df = df.drop(columns=["_id","id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise custom_Exception(e,sys)