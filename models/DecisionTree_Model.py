import numpy as np
import os
import joblib
import pandas as pd
import logging

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SmokingDrinkingTreeTrainer:
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

        feature_names = joblib.load(os.path.join(self.data_folder, 'feature_names.joblib'))

        return X_train, y_train, X_test, y_test, feature_names

    def train(self):
        X_train, y_train, X_test, y_test, feature_names = self.load_data()

        logging.info(f"Setul complet de antrenament are {X_train.shape[0]} rânduri.")

        _, X_search, _, y_search = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )

        logging.info(f"Începem GridSearch-ul pe un eșantion de doar {X_search.shape[0]} rânduri...")

        dtree = DecisionTreeClassifier(random_state=42)

        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced']
        }

        search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        search.fit(X_search, y_search)

        logging.info(f"Căutarea a luat sfârșit! Cei mai buni parametri: {search.best_params_}")
        logging.info("Antrenăm modelul final pe TOT setul de antrenament folosind parametrii optimi...")

        self.best_model = DecisionTreeClassifier(**search.best_params_, random_state=42)
        self.best_model.fit(X_train, y_train)

        logging.info("Antrenament finalizat cu succes!")
        y_pred = self.best_model.predict(X_test)
        logging.info("Evaluare pe setul de test (Multi-Label):")

        target_names = ['is_smoker', 'is_drinker']
        print("\n" + classification_report(y_test, y_pred, target_names=target_names))

        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        joblib.dump(self.best_model, os.path.join(self.results_folder, 'tree_model.joblib'))
        joblib.dump(search.best_params_, os.path.join(self.results_folder, 'tree_best_params.joblib'))
        importance.to_csv(os.path.join(self.results_folder, 'feature_importance.csv'), index=False)

        tree_rules = export_text(self.best_model, feature_names=list(feature_names), max_depth=3)
        with open(os.path.join(self.results_folder, "tree_logic.txt"), "w") as f:
            f.write("Logica Multi-Label Decision Tree (Top 3 Levels):\n")
            f.write(tree_rules)

        logging.info(
            f"Cel mai puternic predictor găsit din date: {importance.iloc[0]['Feature']} (Scor: {importance.iloc[0]['Importance']:.4f})")

        return self.best_model


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    models_dir = os.path.join(ROOT_DIR, 'Models', 'saved_models')

    trainer = SmokingDrinkingTreeTrainer(data_folder=data_dir, results_folder=models_dir)
    try:
        best_tree = trainer.train()
    except Exception as e:
        logging.error(f"Eroare la antrenare: {e}")