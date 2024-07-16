import sys
import os
from dataclasses import dataclass
import numpy as np

#from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Validate data
            logging.info("Validating training and test data")
            X_train, y_train = self._validate_data(X_train, y_train)
            X_test, y_test = self._validate_data(X_test, y_test)


            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                #"K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                #"CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),

            }
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                 "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
                
               
                
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models, param=params)

            ## To get best model score from dist
            best_model_score=max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model= models[best_model_name]

            if best_model_score<.60:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")




            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            R2_square= r2_score(y_test, predicted)

            # Log the best model's R2 score in percentage
            #logging.info(f"Best model R2 score: {R2_square:.2%}")
            
            return  best_model_name, R2_square
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def _validate_data(self, X, y):
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y





        
