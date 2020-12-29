from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineering:
    def __init__(self, train, test, id_column, y_column_name_1,y_column_name_2,y_column_name_3,
                 y_column_name_4,y_column_name_5):
        self.train = train
        self.test = test
        self.y_column_name_1 = y_column_name_1
        self.y_column_name_2 = y_column_name_2
        self.y_column_name_3 = y_column_name_3
        self.y_column_name_4 = y_column_name_4
        self.y_column_name_5 = y_column_name_5
        self.id_column = id_column
        self.data = pd.concat([train, test], ignore_index=True)

    def get_categorical_features(self):
        categorical_features = [feature for feature in self.data.columns if
                                self.data[feature].dtypes == 'O']
        categorical_features_data = self.data[categorical_features]
        return categorical_features_data

    def encode_categorical(self):
        labelEncoder = LabelEncoder()
        categorical_features = self.get_categorical_features()
        mapping_dict = {}
        for feature in categorical_features:
            self.data[feature] = labelEncoder.fit_transform(self.data[feature])
            cat_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
            mapping_dict[feature] = cat_mapping

        with open('../output/dict_house_price.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in mapping_dict.items():
                writer.writerow([key, value])
        return self.data

    def scale_features(self):
        scaler = MinMaxScaler()
        scaling_feature = [feature for feature in self.data.columns if feature not in [self.id_column,
                                                                                       self.y_column_name_1,self.y_column_name_2,
            self.y_column_name_3,self.y_column_name_4,self.y_column_name_5]]
        scaling_features_data = self.data[scaling_feature]
        scale_fit = scaler.fit(scaling_features_data)
        scale_transform = scaler.transform(scaling_features_data)

        data = pd.concat([self.data[[self.id_column, self.y_column_name_1, self.y_column_name_2,
                                     self.y_column_name_3,self.y_column_name_4,self.y_column_name_5]].reset_index(drop=True),
                          pd.DataFrame(scaler.transform(self.data[scaling_feature]), columns=scaling_feature)],
                         axis=1)
        self.data = data
        return self.data