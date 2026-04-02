import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

# Setăm căile
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
models_dir = os.path.join(ROOT_DIR, 'Models', 'saved_models')

# AICI E MODIFICAREA: Am adăugat allow_pickle=True
X_test = np.load(os.path.join(data_dir, 'X_test.npy'), allow_pickle=True)
y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)

target_names = ['is_smoker', 'is_drinker']

# Numele modelelor salvate
model_files = {
    'KNN': 'knn_model.joblib',
    'Decision Tree': 'tree_model.joblib',
    'Naive Bayes': 'bayes_model.joblib'
}

for model_name, file_name in model_files.items():
    print(f"Generăm matricile pentru {model_name}...")
    try:
        # Încărcăm modelul
        model_path = os.path.join(models_dir, file_name)
        model = joblib.load(model_path)
        # Facem predicțiile
        y_pred = model.predict(X_test)

        # Generăm matricea de confuzie multi-label
        mcm = multilabel_confusion_matrix(y_test, y_pred)

        # Desenăm graficele
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Confusion Matrices for {model_name}', fontsize=16)

        for i, (ax, matrix, name) in enumerate(zip(axes, mcm, target_names)):
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['False (0)', 'True (1)'],
                        yticklabels=['False (0)', 'True (1)'])
            ax.set_title(f'Target: {name}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')

        plt.tight_layout()
        # Salvăm imaginea
        plt.savefig(os.path.join(models_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
        plt.show()


    except FileNotFoundError:
        print(f"[Eroare] Modelul {model_name} nu a fost găsit. Ai rulat antrenarea pentru el?")
    except Exception as e:
        print(f"[Eroare] Ceva nu a mers bine la {model_name}: {e}")