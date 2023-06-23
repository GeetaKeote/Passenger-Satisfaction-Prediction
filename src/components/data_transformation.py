import os, sys
import pandas as pd 
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.constant import *
from src.config.configuration import PREPROCESSING_OBJ_PATH,TRANSFORMED_TRAIN_FILE_PATH,TRANSFORMED_TEST_FILE_PATH
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=PREPROCESSING_OBJ_PATH
    transformed_train_path = TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_path = TRANSFORMED_TEST_FILE_PATH

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformation_object(self):
        try:
            logging.info("Loading data transformation")
            

# numarical Columns
            numerical_columns = ['Age', 'Flight Distance', 'Inflight wifi service','Departure/Arrival time convenient',
                                  'Ease of Online booking','Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service','Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes',
       'Arrival Delay in Minutes']
# categorical features
            categorical_columns =['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

            
            
            numerical_pipeline=Pipeline(steps=[
                                ('scaler',StandardScaler()),
                ('transformer', PowerTransformer(method='yeo-johnson', standardize=False))
            ])

            
           

            categorical_pipeline = Pipeline(steps=[
                
                ('ordinal', OrdinalEncoder()),  # Add ordinal encoding step
                ('label', LabelEncoder()),  # Add label encoding step
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])


            preprocessor =ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_columns),
                
                ('category_pipeline',categorical_pipeline,categorical_columns)
            ])

            return preprocessor

            logging.info('pipeline completed')


        except Exception as e:
            logging.info("Error getting data transformation object")
            raise CustomException(e,sys)
        
    def perform_ordinal_encoding(self, df, columns):
        ordinal_encoder = OrdinalEncoder()
        df[columns] = ordinal_encoder.fit_transform(df[columns])
        return df

    def perform_one_hot_encoding(self, df, columns):
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_data = one_hot_encoder.fit_transform(df[columns]).toarray()
        # Create column names for the encoded data
        feature_names = one_hot_encoder.get_feature_names(columns)
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
        df = pd.concat([df, encoded_df], axis=1)
        df.drop(columns=columns, inplace=True, axis=1)
        return df

    def perform_label_encoding(self, df, columns):
        label_encoder = LabelEncoder()
        df[columns] = label_encoder.fit_transform(df[columns])
        return df

    


    def _remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            iqr = Q3 - Q1
            upper_limit = Q3 + 1.5 * iqr
            lower_limit = Q1 - 1.5 * iqr
            df.loc[(df[col]>upper_limit), col]= upper_limit
            df.loc[(df[col]<lower_limit), col]= lower_limit 
            return df
        
        except Exception as e:
            logging.info(" outlier code")
            raise CustomException(e, sys) from e 

          
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')


            logging.info(f"columns in dataframe are: {train_df.columns}")

            logging.info(f"columns in dataframe are: {train_df.dtypes}")

           
            # drop columns
            
            train_df.drop(columns=['Departure Delay in Minutes', 'Arrival Delay in Minutes','Gate location'], inplace=True, axis=1)

            logging.info(f"columns in dataframe are: {train_df.columns}")

            test_df.drop(columns=['Departure Delay in Minutes', 'Arrival Delay in Minutes','Gate location'], inplace=True, axis=1)
            
            logging.info(f"columns in dataframe are: {test_df.columns}")    

            ordinal_encoded_columns = ['Class']
            train_df = self.perform_ordinal_encoding(train_df, ordinal_encoded_columns)
            test_df = self.perform_ordinal_encoding(test_df, ordinal_encoded_columns)

            # perform one-hot encoding
            one_hot_encoded_columns = ['Gender', 'Customer Type', 'Type of Travel']
            train_df = self.perform_one_hot_encoding(train_df, one_hot_encoded_columns)
            test_df = self.perform_one_hot_encoding(test_df, one_hot_encoded_columns)

            # perform label encoding
            label_encoded_columns = ['satisfaction']
            train_df = self.perform_label_encoding(train_df, label_encoded_columns)
            test_df = self.perform_label_encoding(test_df, label_encoded_columns)
       

            
            num_col = [feature for feature in train_df.columns if train_df[feature].dtype != '0']
            
            logging.info(f"numerical_columns: {num_col}")


            cat_col = [feature for feature in train_df.columns if train_df[feature].dtype == 'O']
            logging.info(f"numerical_columns: {cat_col}")


            
            numerical_columns = ['Age', 'Flight Distance', 'Inflight wifi service','Departure/Arrival time convenient',
                                  'Ease of Online booking', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service','Baggage handling', 'Checkin service', 
       'Inflight service','Cleanliness']
            for col in numerical_columns:
                self._remove_outliers_IQR(col=col, df= train_df)

            
            for col in numerical_columns:
                self._remove_outliers_IQR(col=col, df= test_df)
                
            logging.info(f"Outlier capped in test and train df") 



            preprocessing_obj = self.get_data_transformation_object()

            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")



            target_column_name = 'satisfaction'



            X_train = train_df.drop(columns=target_column_name,axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=target_column_name,axis=1)
            y_test = test_df[target_column_name]


            logging.info(f"shape of {X_train.shape} and {y_train.shape}")
            logging.info(f"shape of {X_test.shape} and {y_test.shape}")

            # Transforming using preprocessor obj
            
            X_train=preprocessing_obj.fit_transform(X_train)            
            X_test=preprocessing_obj.transform(X_test)

            logging.info("Applying preprocessing object on training and testing datasets.")
            logging.info(f"shape of {X_train.shape} and {y_train.shape}")
            logging.info(f"shape of {X_test.shape} and {y_test.shape}")
            


            logging.info("transformation completed")



            train_arr =np.c_[X_train,np.array(y_train)]
            test_arr =np.c_[X_test,np.array(y_test)]
            

            logging.info("train arr , test arr")


            df_train= pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            logging.info("converting train_arr and test_arr to dataframe")
            logging.info(f"Final Train Transformed Dataframe Head:\n{df_train.head().to_string()}")
            logging.info(f"Final Test transformed Dataframe Head:\n{df_test.head().to_string()}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path),exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transformed_train_path,index=False,header=True)

            logging.info("transformed_train_path")
            logging.info(f"transformed dataset columns : {df_train.columns}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_path),exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transformed_test_path,index=False,header=True)

            logging.info("transformed_test_path")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor file saved")
            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys) from e 
