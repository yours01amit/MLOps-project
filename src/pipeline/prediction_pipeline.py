import sys
import os
import pandas as pd
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import load_object

# Configure logging
logging.basicConfig(level=logging.INFO)

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Define paths for preprocessor and model
            preprocessor_path = os.path.join("artifacts/pickle", "preprocessor.pkl")
            model_path = os.path.join("artifacts/pickle", "model.pkl")
            
            # Load preprocessor and model
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            # Print features for debugging
            print("Features before transformation:")
            print(features)

            # Scale the data
            scaled_data = preprocessor.transform(features)
            
            # Make prediction
            pred = model.predict(scaled_data)
            
            return pred
            
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):
        
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)

            # Replace empty strings with None (or drop them) instead of 'Unknown'
            df.replace('', None, inplace=True)

            # Check for NaN values and print unique values for debugging
            if df.isnull().values.any():
                logging.warning("Input data contains NaN values.")
            
            logging.info('Dataframe Gathered')
            print("Dataframe content:")
            print(df)
            
            return df
        except Exception as e:
            logging.info('Exception Occurred in gathering data for prediction')
            raise CustomException(e, sys)





# if __name__ == "__main__":
#     # Sample input data for prediction (replace with your actual input)
#     sample_data = CustomData(
#         carat=0.5,
#         depth=60.0,
#         table=55.0,
#         x=4.0,
#         y=4.0,
#         z=2.5,
#         cut='Ideal',
#         color='G',
#         clarity='VS2'
    # )

#     # Convert input data to DataFrame
#     features = sample_data.get_data_as_dataframe()
# 
#     # Create an instance of PredictPipeline
#     pipeline = PredictPipeline()
# 
#     # Make a prediction
#     try:
#         prediction = pipeline.predict(features)
#         print("Prediction:", prediction)
#     except CustomException as e:
#         print("Error during prediction:", e)