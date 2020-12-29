from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import pandas as pd
from processed_data import PreprocessedData
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import pickle

class Model:
    def __init__(self, train, test, id_column, y_column_name_1, y_column_name_2, y_column_name_3,
                 y_column_name_4, y_column_name_5):
        self.label_encoder = LabelEncoder()
        self.y_column_name_1 = y_column_name_1
        self.y_column_name_2 = y_column_name_2
        self.y_column_name_3 = y_column_name_3
        self.y_column_name_4 = y_column_name_4
        self.y_column_name_5 = y_column_name_5
        self.id_column = id_column
        self.number_of_train = train.shape[0]
        self.processed_data = PreprocessedData(train, test, id_column, y_column_name_1,
                            y_column_name_2, y_column_name_3, y_column_name_4, y_column_name_5)
        self.data = self.processed_data.preprocess_my_data()
        self.train = self.data[:self.number_of_train]
        self.test = self.data[self.number_of_train:]
        self.ytrain = self.train[[y_column_name_1, y_column_name_2, y_column_name_3,
                 y_column_name_4, y_column_name_5]]
        self.xtrain = self.train.drop([self.id_column, y_column_name_1, y_column_name_2, y_column_name_3,
                 y_column_name_4, y_column_name_5], axis=1)
        self.xtest = self.test.drop([self.id_column, self.id_column, y_column_name_1, y_column_name_2, y_column_name_3,
                 y_column_name_4, y_column_name_5], axis=1)
        self.clf_models = list()
        self.intiailize_clf_models()

    def get_models(self):

        return self.clf_models

    def add(self, model):

        self.clf_models.append((model))

    def intiailize_clf_models(self):

        model = RandomForestRegressor()
        self.clf_models.append((model))

        model = ExtraTreesRegressor()
        self.clf_models.append((model))

        model = MLPRegressor()
        self.clf_models.append((model))

    def kfold_cross_validation(self):

        clf_models = self.get_models()
        models = []
        self.results = {}

        for model in clf_models:
            self.current_model_name = model.__class__.__name__
            cross_validate = cross_val_score(model, self.xtrain, self.ytrain, cv=4)
            self.mean_cross_validation_score = cross_validate.mean()
            print("Kfold cross validation for", self.current_model_name)
            self.results[self.current_model_name] = self.mean_cross_validation_score
            models.append(model)
            self.save_mean_cv_result()
            print()

    def save_mean_cv_result(self):

        cv_result = pd.DataFrame({'mean_cv_model': self.mean_cross_validation_score}, index=[0])
        file_name = "../output/cv_results/{}.csv".format(self.current_model_name.lower())
        cv_result.to_csv(file_name, index=False)
        print("CV results saved to: ", file_name)

    def show_kfold_cv_results(self):

        for clf_name, mean_cv in self.results.items():
            print("{} cross validation accuracy is {:.3f}".format(clf_name, mean_cv))

    def model_optimization_and_training(self):
        list_of_models = self.get_models()
        rf_model = list_of_models[0]
        #criterion = ['mse', 'mae']
        #num_estimators = [50, 100]
        #parameters = {'n_estimators': num_estimators, 'criterion': criterion}
        #rf_random_search = RandomizedSearchCV(rf_model, param_distributions=parameters)
        fit_model = rf_model.fit(self.xtrain, self.ytrain)
        save_model = pickle.dump(fit_model, open('../models/model_movie_ratings_predictor.pkl', 'wb'))
        return fit_model

    def model_prediction(self):
        rf_model = self.model_optimization_and_training()
        y_predict = rf_model.predict(self.xtest)
        predictions_test_df = pd.DataFrame(data=y_predict, columns=[self.y_column_name_1,self.y_column_name_2,
                                                                    self.y_column_name_3, self.y_column_name_4,
                                                                    self.y_column_name_5])
        return predictions_test_df








