import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import joblib


class SmokingDrinkingPreprocessor:
    def __init__(self, input_path, output_folder):
        self.input_path = input_path
        self.output_folder = output_folder

        # imputers si scaler
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()

        # cream folderul de output daca nu exista
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def run_pipeline(self):
        print(f"Preprocessing: {self.input_path}")

        # citim dataset-ul
        df = pd.read_csv(self.input_path)

        # curatare initiala
        df = df.drop_duplicates()

        # tintele multi-label
        if "DRK_YN" not in df.columns or "SMK_stat_type_cd" not in df.columns:
            raise ValueError("Coloanele tinta 'DRK_YN' sau 'SMK_stat_type_cd' lipsesc")

        # binarizare pentru is_drinker
        df["is_drinker"] = df["DRK_YN"].map({"Y": 1, "N": 0})

        # binarizare pentru is_smoker
        df['is_smoker'] = df['SMK_stat_type_cd'].apply(lambda x: 1 if x == 3 else 0)

        # stratificare combinata (0_0, 1_0, 0_1, 1_1)
        df['stratify_class'] = df['is_smoker'].astype(str) + "_" + df['is_drinker'].astype(str)

        # feature engineering
        if 'sex' in df.columns:
            df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

        if 'weight' in df.columns and 'height' in df.columns:
            df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

        # definirea seturilor X si y
        cols_to_drop = ['DRK_YN', 'SMK_stat_type_cd', 'is_drinker', 'is_smoker', 'stratify_class']
        X = df.drop(columns=cols_to_drop)

        y = df[['is_smoker', 'is_drinker']].values
        stratify_labels = df['stratify_class']

        # split la date 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_labels
        )

        # tratarea coloanelor numerice si categorice
        num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        cat_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()

        # imputare numerica
        if num_cols:
            X_train[num_cols] = self.num_imputer.fit_transform(X_train[num_cols])
            X_test[num_cols] = self.num_imputer.transform(X_test[num_cols])

        # imputare si encoding categorice
        if cat_cols:
            X_train_cat = X_train[cat_cols].astype(object)
            X_test_cat = X_test[cat_cols].astype(object)

            X_train[cat_cols] = self.cat_imputer.fit_transform(X_train_cat)
            X_test[cat_cols] = self.cat_imputer.transform(X_test_cat)

            # one-hot encoding
            X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

            # aliniem testul la train
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # scalarea finala
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # salvam artefactele
        np.save(os.path.join(self.output_folder, 'X_train.npy'), X_train_scaled)
        np.save(os.path.join(self.output_folder, 'X_test.npy'), X_test_scaled)
        np.save(os.path.join(self.output_folder, 'y_train.npy'), y_train)
        np.save(os.path.join(self.output_folder, 'y_test.npy'), y_test)

        joblib.dump(self.scaler, os.path.join(self.output_folder, 'scaler.joblib'))
        joblib.dump(X_train.columns.tolist(), os.path.join(self.output_folder, 'feature_names.joblib'))
        joblib.dump(self.num_imputer, os.path.join(self.output_folder, 'num_imputer.joblib'))

        if cat_cols:
            joblib.dump(self.cat_imputer, os.path.join(self.output_folder, 'cat_imputer.joblib'))

        print(f"Preprocesare completa")
        print(f"Dimensiuni Train: X={X_train_scaled.shape}, y={y_train.shape}")
        print(f"Dimensiuni Test:  X={X_test_scaled.shape}, y={y_test.shape}")


if __name__ == "__main__":
    # calea absoluta
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # caile exacte
    input_file = os.path.join(ROOT_DIR, 'data', 'raw', 'smoking_drinking_dataset.csv')
    output_dir = os.path.join(ROOT_DIR, 'data', 'processed')

    preprocessor = SmokingDrinkingPreprocessor(input_file, output_dir)
    preprocessor.run_pipeline()