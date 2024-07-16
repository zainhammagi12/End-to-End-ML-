import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):

        """This Function is responsible for data transformation"""

        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                     ("scaler",StandardScaler())
                     
                ]

            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                    #("scaler",StandardScaler())

                ]
            )

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")


            logging.info("Numerical column standard scaling completed")
            logging.info("Categorical column encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            #train_df.columns = train_df.columns.str.strip()
            #test_df.columns = test_df.columns.str.strip()

            # Print column names for debugging
            #print("Train DataFrame Columns:", train_df.columns)
            #print("Test DataFrame Columns:", test_df.columns)

            logging.info("Read train and test data completed")

            logging.info("obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformation_object()

            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            traget_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )


            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr= np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]

            test_arr= np.c_[input_feature_test_arr,np.array(traget_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj
                )

            return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                )

        except Exception as e:
            raise CustomException(e,sys)
    