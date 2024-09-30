import os
import sys
from src.logger.logging import logging
from src.exception.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            return train_data_path, test_data_path
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_transformation(self, train_data, test_data):
        try:
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initialize_data_transformation(train_data, test_data)
            return train_arr, test_arr
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_model_training(self, train_arr, test_arr):
        try:
            model_trainer = ModelTrainer()
            model_trainer.initate_model_training(train_arr, test_arr)

        except Exception as e:
            raise CustomException(e, sys)
        



                
    def start_trainig(self):  
        try:
            print("Starting training pipeline...")
            train_data_path, test_data_path = self.start_data_ingestion()
            train_arr, test_arr = self.start_data_transformation(train_data_path, test_data_path)
            self.start_model_training(train_arr, test_arr)

        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.start_trainig() 