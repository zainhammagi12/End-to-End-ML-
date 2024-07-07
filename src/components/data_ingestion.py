import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact',"train.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")
    raw_data_path: str=os.path.join('artifact',"data.csv")



class DataIngesion:
    def __init__(self):
        self.ingesion_connfig=DataIngestionConfig()
    

    def initiate_data_ingesion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r'notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")


            os.makedirs(os.path.dirname(self.ingesion_connfig.test_data_path),exist_ok=True)

            df.to_csv(self.ingesion_connfig.raw_data_path,index=False,header=True)


            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            
            train_set.to_csv(self.ingesion_connfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingesion_connfig.test_data_path,index=False,header=True)


            logging.info("Ingestion of the data completed")

            return(
                self.ingesion_connfig.train_data_path,
                self.ingesion_connfig.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        


if __name__=="__main__":
    obj=DataIngesion()
    train_data, test_data=obj.initiate_data_ingesion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)