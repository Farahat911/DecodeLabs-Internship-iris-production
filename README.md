# Production-Ready Iris Classifier

An enterprise-grade Python application for data classification, built as **Project 2** for the DecodeLabs Artificial Intelligence Internship. This project demonstrates a complete Machine Learning lifecycle, transitioning from basic algorithmic implementations to a robust, deployment-ready architecture.

## 🚀 Key Features

* **Strict Object-Oriented Architecture:** Entire pipeline encapsulated within a highly cohesive Python class with explicit type hinting for maintainability.
* **Leak-Proof Sklearn Pipeline:** Integrates `StandardScaler` and `KNeighborsClassifier` into a unified `sklearn.pipeline.Pipeline`, ensuring zero data leakage during cross-validation.
* **Advanced Hyperparameter Tuning:** Replaces hardcoded values with dynamic optimization. Uses `GridSearchCV` to evaluate multiple `n_neighbors` (odd numbers to prevent ties) and distance metrics (`uniform` vs `distance`) via 5-fold cross-validation.
* **Overfitting Prevention:** Starts neighbor search from $K=3$ to avoid the noise and overfitting commonly associated with $K=1$.
* **Stratified Splitting:** Employs `stratify=y` during the train/test split to guarantee representative class distribution.
* **Model Serialization:** Automatically exports the optimized, final pipeline using `joblib`, rendering the model fully prepared for API deployment.

## 🛠️ Tech Stack

* **Language:** Python 3
* **Machine Learning Framework:** scikit-learn
* **Data Processing:** NumPy
* **Serialization:** joblib

## 📂 Project Structure

├── iris_production.py
├── iris_production_model.joblib (Generated upon execution)
└── README.md

## ⚙️ Installation & Usage

1. **Clone the repository:**
   git clone <your-repository-url>
   cd DecodeLabs-Internship/Project_2

2. **Install dependencies:**
   Ensure you have scikit-learn and numpy installed:
   pip install scikit-learn numpy joblib

3. **Run the script:**
   python iris_production.py

4. **Output:** The script will output the optimal parameters, the cross-validation F1-score, the test set confusion matrix, and finally, generate the `.joblib` file.

## 📝 License

This project was developed for the DecodeLabs Industrial Training Kit (Batch 2026).
