import argparse
import pickle

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

import config
from dataset import load_data, scale_data, separate_data
from features import select_features

from plots import (
    confusion_matrix_plot,
    correlation_matrix,
    curve_comparison,
    f1_comparison_bar,
    feature_importance_plot,
)
from predict import evaluate_models


def build_pipline(classifier, RANDOM_STATE):
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', classifier)])
    return pipeline


def train_model(pipeline, param_grid, model_path, X_train, y_train, n_iter, random_state):

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=5,
        n_jobs=-1,
        scoring='f1',
        random_state=random_state)
    search.fit(X_train, y_train)

    with open(model_path, 'wb') as file:
        pickle.dump(search.best_estimator_, file)

    return search.best_estimator_


def train_all_models(X_train, y_train, n_iter=config.SEARCH_N_ITER):
    models = {}
    for name, classifier in config.CLASSIFIERS.items():
        print(f"Training {name}...")
        pipeline = build_pipline(classifier, config.RANDOM_STATE)
        models[name] = train_model(
            pipeline,
            config.PARAM_GRIDS[name],
            config.MODEL_PATHS[name],
            X_train,
            y_train,
            n_iter,
            config.RANDOM_STATE,
        )
    return models


def main():

    parser = argparse.ArgumentParser(description="Data Science Pipeline")

    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "plot"],
        required=True,
        help="train = train models, evaluate = evaluate saved models, plot = generate visualizations"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=config.SEARCH_N_ITER,
        help="number of parameter settings sampled by RandomizedSearchCV per model (train mode only)"
    )

    args = parser.parse_args()

    #Load Data
    df = load_data(config.DATA_PATH)
    #Feature Selection
    X, y = select_features(df)
    #Make training/testing datasets
    X_train, X_test, y_train, y_test = separate_data(X,
                                                     y,
                                                     config.TEST_SIZE,
                                                     config.RANDOM_STATE)
    #Scale features
    X_train, X_test = scale_data(X_train, X_test)

    #Train Models
    if args.mode == "train":
        train_all_models(X_train, y_train, args.n_iter)

    #Load Models
    elif args.mode == "evaluate":
        results = evaluate_models(X_test, y_test, config.MODEL_PATHS)
        print(results)

    #Plot models (found in models directory)
    elif args.mode == "plot":
        #Save correlation Matrix
        correlation_matrix(df)

        #Highlight top performing models: random forest and xgboost
        top_models = {"rf": config.MODEL_PATHS["rf"], "xgb": config.MODEL_PATHS["xgb"]}
        for name, model_path in top_models.items():
            feature_importance_plot(model_path, X, name)
            confusion_matrix_plot(model_path, X_test, y_test, name)
        curve_comparison(top_models, X_test, y_test)

        #F1 comparison across all trained models, with rf/xgb called out
        results = evaluate_models(X_test, y_test, config.MODEL_PATHS)
        f1_comparison_bar(results, highlight=tuple(top_models))

if __name__ == "__main__":
    main()
