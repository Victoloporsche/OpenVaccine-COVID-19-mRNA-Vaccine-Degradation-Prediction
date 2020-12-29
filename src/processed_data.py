import pandas as pd
from feature_engineering import FeatureEngineering

class PreprocessedData:
    def __init__(self, train, test, id_column, y_column_name_1, y_column_name_2, y_column_name_3,
                 y_column_name_4, y_column_name_5):
        self.train = train
        self.test = test
        self.y_column_name_1 = y_column_name_1
        self.y_column_name_2 = y_column_name_2
        self.y_column_name_3 = y_column_name_3
        self.y_column_name_4 = y_column_name_4
        self.y_column_name_5 = y_column_name_5
        self.id_column = id_column
        self.data = pd.concat([train, test], ignore_index=True)
        self.feature_engineering = FeatureEngineering(train, test, id_column,
                            y_column_name_1, y_column_name_2, y_column_name_3,y_column_name_4, y_column_name_5)


    def preprocess_my_data(self):
        self.data = self.feature_engineering.encode_categorical()
        self.data = self.feature_engineering.scale_features()
        return self.data
