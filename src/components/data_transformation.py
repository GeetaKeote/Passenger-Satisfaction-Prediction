import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.Utils.utils import save_object
from sklearn.base import BaseEstimator, TransformerMixin

@dataclass
class DataTransfromartionConfigs:
    preprocess_obj_file_patrh = os.path.join("artifacts/data_transformation", "preprcessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransfromartionConfigs()

    def delete_columns(self, columns_to_delete, df):
        try:
            df = df.drop(columns_to_delete, axis=1)
            return df
        except Exception as e:
            logging.info("Error occurred while deleting columns")
            raise CustomException(e, sys)
    
    def remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            iqr = Q3 - Q1

            upper_limit = Q3 + 1.5 * iqr
            lowwer_limit = Q1 - 1.5 * iqr

            df.loc[(df[col]>upper_limit), col] = upper_limit
            df.loc[(df[col]<lowwer_limit), col] = lowwer_limit

            return df

        except Exception as e:
            logging.info("Outliers handling code")
            raise CustomException(e, sys)
        
    

    def get_data_transformation_obj(self):
        try:

            logging.info(" Data Transformation Started")

            numerical_features = [ 'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                                   'Ease of Online booking', 'Food and drink', 'Online boarding', 'Seat comfort', 
                                   'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 
                                   'Checkin service', 'Inflight service', 'Cleanliness' ]

            #numerical_features =['Unnamed: 0', 'Age', 'Flight Distance', 'Inflight wifi service',
                                 # 'Departure/Arrival time convenient', 'Ease of Online booking', 
                                #  'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 
                                 # 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 
                                #  'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
            
            ordinal_columns = ['Class']
            onehot_columns = ['Gender', 'Customer Type', 'Type of Travel' ]           
            #label_encoder_column = ['satisfaction']
            #class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
                #def __init__(self):
                  #  self.label_encoder = LabelEncoder()
                
                #def fit(self, X, y=None):
                  #  self.label_encoder.fit(X)
                    #return self
                
                #def transform(self, X):
                    #return self.label_encoder.transform(X).reshape(-1, 1)

            
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
            )


            onehot_pipeline= Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('oridnal_encoder', OrdinalEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])


            ordinal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])
            #label_encoder_pipeline = Pipeline(steps=[
                #('label_encoder', LabelEncoder())
            #])
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('onehot_pipeline', onehot_pipeline, onehot_columns),
                ('ordinal_pipeline', ordinal_pipeline, ordinal_columns),
                #('label_encoder_pipeline', label_encoder_pipeline, label_encoder_column)
            ])
            
       
            return preprocessor
        
      



        except Exception as e:
            raise CustomException(e, sys)
        
    
        
    def inititate_data_transformation(self, train_path, test_path):

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            

            print(train_data.columns)
            print(test_data.columns)

            train_data.columns = train_data.columns.str.strip()
            test_data.columns = test_data.columns.str.strip()

            # Map values for the 'satisfaction' column
            satisfaction_mapping = {
                'satisfied': 1,
                'neutral or dissatisfied': 0
            }
            train_data['satisfaction'] = train_data['satisfaction'].map(satisfaction_mapping)
            test_data['satisfaction'] = test_data['satisfaction'].map(satisfaction_mapping)

            # Columns to delete
            columns_to_delete = ['Unnamed: 0','Gate location',]

            # Delete columns from train_data and test_data
            train_data = self.delete_columns(columns_to_delete, train_data)
            test_data = self.delete_columns(columns_to_delete, test_data)
            


            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')


            logging.info(f"columns in dataframe are: {train_data.columns}")
            logging.info(f"Catogorical columns in dataframe are: {train_data.select_dtypes(include=['object']).columns.tolist()}")
            logging.info(f"Numerical columns in dataframe are: {train_data.select_dtypes(include=['int64','float64']).columns.tolist()}")
            logging.info(f"columns in dataframe are: {test_data.dtypes}")

            categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'satisfaction','Departure Delay in Minutes', 'Arrival Delay in Minutes']
            missing_columns = []

            for col in categorical_columns:
                if col not in train_data.columns:
                    missing_columns.append(col)

            if missing_columns:
                raise ValueError(f"The following categorical columns are missing from the dataset: {missing_columns}")
            
            

            

            numerical_features = [ 'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                                   'Ease of Online booking', 'Food and drink', 'Online boarding', 'Seat comfort', 
                                   'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 
                                   'Checkin service', 'Inflight service', 'Cleanliness' ]
        

            for col in numerical_features:
                self.remove_outliers_IQR(col = col, df = train_data)
                self.remove_outliers_IQR(col = col, df = test_data)
            logging.info("Outliers capped on our train data")

              
            
            preprocess_obj = self.get_data_transformation_obj()

            traget_columns =['satisfaction']
            drop_columns =traget_columns

            

            logging.info("Splitting train data into dependent and independent features")
            input_feature_train_data = train_data.drop(drop_columns, axis = 1)
            traget_feature_train_data = train_data[traget_columns]

            logging.info("Splitting test data into dependent and independent features")
            input_feature_test_data = test_data.drop(drop_columns, axis = 1)
            traget_feature_test_data = test_data[traget_columns]

            # Apply transfpormation on our train data and test data
            input_train_arr = preprocess_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocess_obj.transform(input_feature_test_data)

            # Apply preprocessor object on our train data and test data
            train_array = np.c_[input_train_arr, np.array(traget_feature_train_data)]
            test_array = np.c_[input_test_arr, np.array(traget_feature_test_data)]


            save_object(file_path=self.data_transformation_config.preprocess_obj_file_patrh,
                        obj=preprocess_obj)
            logging.info("Data Transformation Completed")
            return (train_array,
                    test_array,
                    self.data_transformation_config.preprocess_obj_file_patrh)



        except Exception as e:
            raise CustomException(e, sys)
