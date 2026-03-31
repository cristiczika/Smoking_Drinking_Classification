import numpy as np
import os
import joblib
import pandas as pd
import logging

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Setăm configurarea pentru afișarea jurnalelor (logs)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SmokingDrinkingTreeTrainer:
    def __init__(self, data_folder, results_folder):
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.best_model = None

        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def load_data(self):
        # Încărcăm datele salvate anterior din format Numpy
        X_train = np.load(os.path.join(self.data_folder, 'X_train.npy'))
        y_train = np.load(os.path.join(self.data_folder, 'y_train.npy'))
        X_test = np.load(os.path.join(self.data_folder, 'X_test.npy'))
        y_test = np.load(os.path.join(self.data_folder, 'y_test.npy'))

        # Încărcăm numele coloanelor pentru a le avea în analiza finală
        feature_names = joblib.load(os.path.join(self.data_folder, 'feature_names.joblib'))

        # ATENȚIE: Aici NU mai folosim .ravel() cum era în original.
        # Într-o problemă Multi-Label, y_train și y_test trebuie să rămână matrici 2D (Nx2)
        return X_train, y_train, X_test, y_test, feature_names

    def train(self):
        X_train, y_train, X_test, y_test, feature_names = self.load_data()
        logging.info(f"Antrenare Multi-Label Decision Tree pe {X_train.shape[1]} variabile...")
        logging.info(f"Se prezic simultan {y_train.shape[1]} etichete: [Fumător, Băutor]")

        # DecisionTreeClassifier din scikit-learn suportă nativ multi-output!
        dtree = DecisionTreeClassifier(random_state=42)

        # Grila de parametri (Hiperparametri) pentru optimizare
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced']
        }

        # Grid Search cu Cross-Validation (cv=5)
        # Folosește toate core-urile procesorului (n_jobs=-1) pentru viteză
        search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)

        # Extragem cel mai bun model găsit de GridSearch
        self.best_model = search.best_estimator_

        # Evaluare pe setul de test
        y_pred = self.best_model.predict(X_test)
        logging.info("Evaluare pe setul de test (Multi-Label):")

        # Generăm raportul specificând numele celor 2 ținte pentru claritate
        target_names = ['is_smoker', 'is_drinker']
        print("\n" + classification_report(y_test, y_pred, target_names=target_names))

        # Chiar dacă este multi-output, arborele calculează o importanță unificată pentru fiecare trăsătură
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Salvăm rezultatele
        joblib.dump(self.best_model, os.path.join(self.results_folder, 'tree_model.joblib'))
        joblib.dump(search.best_params_, os.path.join(self.results_folder, 'tree_best_params.joblib'))
        importance.to_csv(os.path.join(self.results_folder, 'feature_importance.csv'), index=False)

        # Exportăm logica arborelui (primele 3 nivele pentru a nu genera un fișier uriaș)
        tree_rules = export_text(self.best_model, feature_names=list(feature_names), max_depth=3)
        with open(os.path.join(self.results_folder, "tree_logic.txt"), "w") as f:
            f.write("Logica Multi-Label Decision Tree (Top 3 Levels):\n")
            f.write(tree_rules)

        logging.info(f"Cei mai buni hiperparametri găsiți: {search.best_params_}")
        logging.info(
            f"Cel mai puternic predictor găsit din date: {importance.iloc[0]['Feature']} (Scor: {importance.iloc[0]['Importance']:.4f})")

        return self.best_model


if __name__ == "__main__":
    # Calculăm folderul rădăcină (adaptează dacă este nevoie)
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Căile conform structurii de fișiere rezultate din preprocesare
    data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    models_dir = os.path.join(ROOT_DIR, 'Models', 'saved_models')

    trainer = SmokingDrinkingTreeTrainer(data_folder=data_dir, results_folder=models_dir)
    try:
        best_tree = trainer.train()
    except Exception as e:
        logging.error(f"Eroare la antrenare: {e}")