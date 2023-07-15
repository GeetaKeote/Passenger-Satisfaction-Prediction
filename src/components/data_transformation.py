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
from src.config.configuration import PREPROCESSING_OBJ_PATH,TRANSFORMED_TRAIN_FILE_PATH,TRANSFORMED_TEST_FILE_PATH,FEATURE_ENG_OBJ_PATH

class Feature_Engineering(BaseEstimator,TransformerMixin):
    def __init__(self):   #class to apply Feature Emggneering
         logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")


    def _remove_outliers_IQR(self,col,df):
        try:
            Q1=df[col].quantile(0.25)
            Q3=df[col].quantile(0.75)
            iqr=Q3-Q1
            upper_limit = Q3+1.5 *iqr
            lower_limit =Q1 -1.5*iqr
            df.loc[(df[col]>upper_limit),col]=upper_limit
            df.loc[(df[col]<lower_limit),col]=lower_limit
            return df             

        except Exception as e:
            logging.info("outlier code removal by IQR Method")
            raise CustomException(e,sys)
        
    def transform_data(self,df):
        try:
            num_col = [feature for feature in df.columns if df[feature].dtype != '0']
            
            logging.info(f"numerical_columns: {num_col}")


            cat_col = [feature for feature in df.columns if df[feature].dtype == 'O']
            logging.info(f"categorical_columns: {cat_col}")

            #df.drop(columns=['Unnamed: 0','Gate location','Departure Delay in Minutes', 'Arrival Delay in Minutes'], inplace=True, axis=1)

            logging.info(f"columns in dataframe are: {df.columns}")

            numerical_columns = [ 'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                                   'Ease of Online booking', 'Food and drink', 'Online boarding', 'Seat comfort', 
                                   'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 
                                   'Checkin service', 'Inflight service', 'Cleanliness' ]


# outlier

            for col in numerical_columns:
                self._remove_outliers_IQR(col=col, df= df)
            
            logging.info(f"Outlier capped in train df")
            return df 
            
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            transformed_df=self.transform_data(X)
                
            return transformed_df
        except Exception as e:
            raise CustomException(e,sys) from e
        
        


@dataclass
class DataTransfromartionConfigs:
   # preprocess_obj_file_patrh = os.path.join("artifacts/data_transformation", "preprcessor.pkl")
    preprocessor_obj_file_path=PREPROCESSING_OBJ_PATH
    transformed_train_path=TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_path=TRANSFORMED_TEST_FILE_PATH
    feature_eng_obj_path=FEATURE_ENG_OBJ_PATH


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
    
   
        
    

    def get_data_transformation_obj(self):
        try:

            logging.info(" Data Transformation Started")

            numerical_features = [ 'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                                   'Ease of Online booking', 'Food and drink', 'Online boarding', 'Seat comfort', 
                                   'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 
                                   'Checkin service', 'Inflight service', 'Cleanliness' ]

            
            
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
                ('oridnal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                ('scaler', StandardScaler(with_mean=False))
            ])


            ordinal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
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
        
    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering())])
            return feature_engineering
        except Exception as e:
            raise CustomException(e,sys) from e
    
        
    def inititate_data_transformation(self, train_path, test_path):

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            

            print(train_data.columns)
            print(test_data.columns)

            train_data.columns = train_data.columns.str.strip()
            test_data.columns = test_data.columns.str.strip()

            train_data = train_data.dropna()
            test_data = test_data.dropna()

            # Map values for the 'satisfaction' column
            satisfaction_mapping = {
                'satisfied': 1,
                'neutral or dissatisfied': 0
            }
            train_data['satisfaction'] = train_data['satisfaction'].map(satisfaction_mapping)
            test_data['satisfaction'] = test_data['satisfaction'].map(satisfaction_mapping)

            # Columns to delete
            columns_to_delete = ['Unnamed: 0','Gate location','Departure Delay in Minutes', 'Arrival Delay in Minutes']

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

            categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'satisfaction']
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
            
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
        

            
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")
            train_data = fe_obj.fit_transform(train_data)
            logging.info(">>>" * 20 + " Test data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Test Data ")
            test_data = fe_obj.transform(test_data)

            if train_data is not None:
                train_data.to_csv("train_data.csv")
                logging.info(f"Saving csv to train_data.csv")

            if test_data is not None:
                test_data.to_csv("test_data.csv")
                logging.info(f"Saving csv to test_data.csv")


           #train_data.to_csv("train_data.csv")
            #test_data.to_csv("test_data.csv")
            logging.info(f"Saving csv to train_data and test_data.csv")

            # Preprocessing pipeline                         
            
            preprocess_obj = self.get_data_transformation_obj()

            target_column_name ='satisfaction' 
            logging.info(f"shape of {train_data.shape} and {test_data.shape}")
            

            X_train = train_data.drop(columns=target_column_name,axis=1)
            y_train=train_data[target_column_name]

            X_test=test_data.drop(columns=target_column_name,axis=1)
            y_test=test_data[target_column_name]

            logging.info(f"shape of {X_train.shape} and {X_test.shape}")
            logging.info(f"shape of {y_train.shape} and {y_test.shape}")

            X_train=preprocess_obj.fit_transform(X_train)            
            X_test=preprocess_obj.transform(X_test)
            logging.info("Applying preprocessing object on training and testing datasets.")
            logging.info(f"shape of {X_train.shape} and {X_test.shape}")
            logging.info(f"shape of {y_train.shape} and {y_test.shape}")
            

            logging.info("transformation completed")

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            logging.info("train_arr, test_arr completed")

            

            logging.info("train arr , test arr")


            train_data= pd.DataFrame(train_arr)
            test_data = pd.DataFrame(test_arr)

            logging.info("converting train_arr and test_arr to dataframe")


            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path),exist_ok=True)
            train_data.to_csv(self.data_transformation_config.transformed_train_path,index=False,header=True)

            logging.info("transformed_train_path")
            logging.info(f"transformed dataset columns : {train_data.columns}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_path),exist_ok=True)
            test_data.to_csv(self.data_transformation_config.transformed_test_path,index=False,header=True)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocess_obj)
            
            logging.info("Preprocessor file saved")


            save_object(
                file_path=self.data_transformation_config.feature_eng_obj_path,
                obj=fe_obj)
            logging.info("Feature eng file saved")
            logging.info("Data Transformation Completed")
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
