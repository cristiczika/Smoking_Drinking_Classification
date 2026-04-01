import numpy as np
import os
import joblib
import logging

from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SmokingDrinkingBayesTrainer:
    def __init__(self, data_folder, results_folder):
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.best_model = None

        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def load_data(self):
        X_train = np.load(os.path.join(self.data_folder, 'X_train.npy'))
        y_train = np.load(os.path.join(self.data_folder, 'y_train.npy'))
        X_test = np.load(os.path.join(self.data_folder, 'X_test.npy'))
        y_test = np.load(os.path.join(self.data_folder, 'y_test.npy'))

        return X_train, y_train, X_test, y_test

    def train(self):
        X_train, y_train, X_test, y_test = self.load_data()

        logging.info(f"Setul complet de antrenament are {X_train.shape[0]} rânduri.")
        logging.info("Începem GridSearch pentru Gaussian Naive Bayes...")

        base_model = MultiOutputClassifier(GaussianNB())

        param_grid = {
            'estimator__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }

        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1_micro',
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        logging.info(f"Căutarea a luat sfârșit! Cei mai buni parametri: {search.best_params_}")
        logging.info("Antrenăm modelul final pe TOT setul de antrenament folosind parametrul optim...")

        self.best_model = search.best_estimator_
        self.best_model.fit(X_train, y_train)

        logging.info("Antrenament finalizat cu succes!")

        y_pred = self.best_model.predict(X_test)

        logging.info("Evaluare pe setul de test (Multi-Label):")

        target_names = ['is_smoker', 'is_drinker']
        print("\n" + classification_report(y_test, y_pred, target_names=target_names))

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score (macro): {f1_macro:.4f}")

        joblib.dump(self.best_model, os.path.join(self.results_folder, 'bayes_model.joblib'))
        joblib.dump(search.best_params_, os.path.join(self.results_folder, 'bayes_best_params.joblib'))

        with open(os.path.join(self.results_folder, "bayes_metrics.txt"), "w") as f:
            f.write("Gaussian Naive Bayes - Multi-Label Evaluation\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(y_test, y_pred, target_names=target_names))
            f.write("\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1-score (macro): {f1_macro:.4f}\n")
            f.write(f"Best Params: {search.best_params_}\n")

        logging.info("Modelul și metricile au fost salvate cu succes.")

        return self.best_model


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    models_dir = os.path.join(ROOT_DIR, 'Models', 'saved_models')

    trainer = SmokingDrinkingBayesTrainer(data_folder=data_dir, results_folder=models_dir)

    try:
        best_bayes = trainer.train()
    except Exception as e:
        logging.error(f"Eroare la antrenare: {e}")