import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from typing import Dict, Any, Tuple, List
import joblib
import os


class IrisProductionClassifier:
    def __init__(self, test_size: float = 0.2, random_state: int = 42, cv_folds: int = 5) -> None:
        self.test_size: float = test_size
        self.random_state: int = random_state
        self.cv_folds: int = cv_folds
        self.X_train: np.ndarray = np.array([])
        self.X_test: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.y_test: np.ndarray = np.array([])
        self.pipeline: Pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])
        self.grid_search: GridSearchCV = GridSearchCV(
            estimator=self.pipeline,
            param_grid={
                'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'knn__weights': ['uniform', 'distance']
            },
            cv=self.cv_folds,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = 0.0
        self.best_estimator_: Pipeline = self.pipeline

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        data = load_iris()
        return data.data, data.target

    def split_data(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=y
        )

    def tune_hyperparameters(self) -> None:
        self.grid_search.fit(self.X_train, self.y_train)
        self.best_params_ = self.grid_search.best_params_
        self.best_score_ = self.grid_search.best_score_
        self.best_estimator_ = self.grid_search.best_estimator_

    def evaluate(self) -> Dict[str, Any]:
        predictions = self.best_estimator_.predict(self.X_test)
        cm = confusion_matrix(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average='weighted')
        return {'confusion_matrix': cm, 'f1_score': f1}

    def export_model(self, filepath: str = 'iris_production_model.joblib') -> str:
        joblib.dump(self.best_estimator_, filepath)
        return os.path.abspath(filepath)

    def run(self) -> Dict[str, Any]:
        X, y = self.load_data()
        self.split_data(X, y)
        self.tune_hyperparameters()
        results = self.evaluate()
        results['best_params'] = self.best_params_
        results['cv_score'] = self.best_score_
        return results


def main() -> None:
    classifier = IrisProductionClassifier()
    results = classifier.run()

    print("Optimal Parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")

    print(f"\nCross-Validation Weighted F1-Score: {results['cv_score']:.4f}")

    print("\nConfusion Matrix:")
    for row in results['confusion_matrix']:
        print(" ".join(str(val).rjust(3) for val in row))

    print(f"\nTest Set Weighted F1-Score: {results['f1_score']:.4f}")

    export_path = classifier.export_model()
    print(f"\nModel exported to: {export_path}")


if __name__ == "__main__":
    main()
