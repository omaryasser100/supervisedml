import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, train_path):
        # Convert to Path object and resolve relative to project root
        self.train_path = Path(__file__).parent / train_path
        
    def load_data(self):
        """Load the training data"""
        try:
            return pd.read_csv(self.train_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find data file at: {self.train_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def split(self, df, test_size=0.2, random_state=42):
        """Split data into training and test sets"""
        X = df.drop('SalePrice', axis=1)
        y = df['SalePrice']
        return train_test_split(X, y, test_size=test_size, random_state=random_state)