import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.Utils.utils import save_object,evaluate_model
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

@dataclass
class ModelTrainerConfig:
    train_model_filepath= os.path.join("artifacts/model_trainer", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

        logging.info("Model Training Started")

    def inititate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spitting our dataset into Dependebt and independent features")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            model={
                "Logistic":LogisticRegression(),
                 "DecisionTree":DecisionTreeClassifier(),
                 "Gradient Booasting":XGBClassifier(),
                 'XGB Classifier':XGBClassifier(),
                 'KNN neighbour':KNeighborsClassifier(),
                 "Random Forest":RandomForestClassifier()
            }            
            params={
                "Logistic":{
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [100, 500, 2000]
                },
                 "DecisionTree":{
                     "class_weight":["balanced"],
                     "criterion":['gini',"entropy","log_loss"],
                     "splitter":['best','random'],
                     "max_depth":[3,4,5,6],
                     "min_samples_split":[2,3,4,5],
                     "min_samples_leaf":[1,2,3],
                     "max_features":["auto","sqrt","log2"],
                     "max_features":["sqrt"]
                     },
                    
                 "Gradient Booasting":{
                     "learning_rate":[ 0.1, 0.05],
                     "n_estimators":[50,100],
                     "max_depth":[10, 8 ]
                 },
                 'XGB Classifier':{
                     'max_depth': [ 5, 7],
                     'learning_rate': [0.1, 0.01],
                     'n_estimators': [100, 300],
                     'colsample_bytree': [0.8], 
                     'n_jobs':[-1],
                     'reg_alpha': [ 0.1, 0.5],
                     'reg_lambda': [ 1, 10]
                 },
                 "KNN neighbour":{
                     'n_neighbors': [2, 5, 7],
                         'weights': ['uniform', 'distance'],
                 },
                 "Random Forest":{
                     'n_estimators': [500,  30,50,100],
                     'max_depth': [10, 8, 5,None],
                     'min_samples_split': [2, 5, 8],
                     'criterion':["gini"]
                 }
            }

            model_report:dict=evaluate_model(X_train =X_train ,X_test=X_test,y_train=y_train,y_test=y_test,
                                           models=model,params=params)
            # to get the best Model from Report
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model.keys())[
               list(model_report.values()).index(best_model_score)
            ]
            best_model=model[best_model_name]
            #best_model_name = None
            #best_model_score = 0.0

            #for model_name, model in model.items():
               #param_grid = params[model_name]

                #grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
                #grid_search.fit(X_train, y_train)

                # Get the best model from GridSearchCV
                #if grid_search.best_score_ > best_model_score:
                  #  best_model = grid_search.best_estimator_
                   # best_model_score = grid_search.best_score_

          
                  

            print(f"Best Model Found,Model is:{best_model_name},Accuracy_Score:{best_model_score}")
            print("\n------------------------------------------------------------------------------")
            logging.info(f"Best Model Found,Model is:{best_model_name},Accuracy_Score:{best_model_score}")

            save_object(file_path=self.model_trainer_config.train_model_filepath,obj=best_model)


        except Exception as e:
            raise Exception(e,sys)