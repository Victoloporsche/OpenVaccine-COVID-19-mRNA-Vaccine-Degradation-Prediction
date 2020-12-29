from model import Model

class Main:
    def __init__(self, train, test, id_column, y_column_name_1, y_column_name_2, y_column_name_3,
                 y_column_name_4, y_column_name_5):
        self.y_column_name_1 = y_column_name_1
        self.y_column_name_2 = y_column_name_2
        self.y_column_name_3 = y_column_name_3
        self.y_column_name_4 = y_column_name_4
        self.y_column_name_5 = y_column_name_5
        self.id_column = id_column
        self.model = Model(train, test, id_column, y_column_name_1, y_column_name_2, y_column_name_3,
                 y_column_name_4, y_column_name_5)

    def cross_validation(self):
        return self.model.kfold_cross_validation()

    def show_cross_validation_result(self):
        return self.model.show_kfold_cv_results()

    def model_training(self):
        return self.model.model_optimization_and_training()

    def model_prediction(self):
        return self.model.model_prediction()