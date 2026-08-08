from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = "data/raw/term-deposit-marketing-2020.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SEARCH_N_ITER = 20  # RandomizedSearchCV draws; capped automatically for smaller grids

MODEL_PATHS = {
    "rf": "models/rf_model.pkl",
    "lr": "models/lr_model.pkl",
    "knn": "models/knn_model.pkl",
    "svm": "models/svm_model.pkl",
    "xgb": "models/xgb_model.pkl",
    "dt": "models/dt_model.pkl",
}

CLASSIFIERS = {
    "rf": RandomForestClassifier(random_state=RANDOM_STATE),
    "lr": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
    "knn": KNeighborsClassifier(),
    "svm": SVC(random_state=RANDOM_STATE),
    "xgb": XGBClassifier(objective="binary:logistic", random_state=RANDOM_STATE),
    "dt": DecisionTreeClassifier(random_state=RANDOM_STATE),
}

# Same search spaces used for hyperparameter tuning in notebooks/modeling.ipynb
PARAM_GRIDS = {
    "rf": {
        "classifier__n_estimators": [100, 200, 400],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    },
    "lr": {
            "classifier__penalty": ["l1", "l2"],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["liblinear", "lbfgs"],
        },
    "knn": {
        "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2],
    },
    "svm": {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale", "auto"],
    },
    "xgb": {
        "classifier__max_depth": [3, 5, 7],
        "classifier__min_child_weight": [1, 3, 5],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__learning_rate": [0.01, 0.1, 0.3],
        "classifier__n_estimators": [50, 100, 200],
    },
    "dt": {
        "classifier__max_depth": [3, 5, 10, 20, None],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__criterion": ["gini", "entropy"],
    },
}
