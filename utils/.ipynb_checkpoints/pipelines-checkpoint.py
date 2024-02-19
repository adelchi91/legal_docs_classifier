import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.feature_selection import ColumnSelector
# from sklearn.compose import ColumnTransformer


# import optbinning
import numpy as np

# Column selector
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.cols]

 
# features_covariate = ["Temperature",  "Humidity", "Light", "CO2", "HumidityRatio"]

# clf_v3_optb_logit = sklearn.pipeline.Pipeline(
#     steps=[
#         (
#             "encoder",
#             optbinning.BinningProcess(
#                 variable_names=features_covariate,
#                 # categorical_variables=v3_cat_features,
#                 max_pvalue=0.05,
#             ),
#         ),
#         # ("removing_features_with_low_variance", VarianceThreshold(threshold=0.01)),
#         ("scaler", sklearn.preprocessing.StandardScaler()),
#         # ('feature_selection', feature_selector),
#         (
#             "logistic",
#             sklearn.linear_model.LogisticRegression(random_state=0),
#         ),
#     ]
# )
# clf_v3_optb_logit.version = "v3.1-optb-logit"

clf_rf = sklearn.pipeline.Pipeline(
    steps=[
        (
            "vectorization",
            TfidfVectorizer(max_features=5000),
        ),
        (
            "forest",
            RandomForestClassifier(random_state=0),
        ),
    ]
)
clf_rf.version = "rf_classifier"

clf_adaboost = sklearn.pipeline.Pipeline(
    steps=[
        ('vectorization', TfidfVectorizer(max_features=4000)),
        ("adaboost ", AdaBoostClassifier(n_estimators=50, random_state=42)),
    ]
)

# clf_adaboost = Pipeline([
#     ('features', ColumnTransformer([
#         ('text', TfidfVectorizer(max_features=5000), 'Cleaned_Content')
#     ], remainder='passthrough')),
#     ('adaboost', AdaBoostClassifier(n_estimators=50, random_state=42)),
# ])
clf_adaboost.version = "adaboost_classifier"


# grid of hyperparameters to be tested
grid = [
    # {
    #     "clf": [clf_v3_optb_logit],
    #     # "clf__encoder__binning_fit_params": { [None, 0.01, 0.02, 0.03, 0.04, 0.05]},
    #     # "clf__logistic__C": [0.01, 0.03, 0.1, 0.3, 1],
    #     "clf__logistic__C": np.logspace(-3, 1, 10),  # np.linspace(0.001, 1, 10), # np.logspace(-3, 1, 5),
    #     "clf__logistic__solver": ["saga"],
    #     "clf__logistic__penalty": ["l1", "l2"],  # "elasticnet"],
    #     # "clf__logistic__l1_ratio": [0.0, 0.01, 0.1, 0.2, 0.5, 0.8, 1.0],
    #     "clf__logistic__max_iter": [5000],
    # },
    # {
    #     "clf": [clf_rf],
    #     "clf__forest__n_estimators": [50, 100, 200],  # Add RandomForest hyperparameters
    #     "clf__forest__max_depth": [None, 10, 20],
    #     "clf__forest__min_samples_split": [2, 5, 10],
    #     "clf__forest__min_samples_leaf": [1, 2, 4],
    # },

    {
        "clf": [clf_adaboost],
        "clf__adaboost__n_estimators": [50],#, 100],#, 200],  # Add AdaBoost hyperparameters
        # "clf__adaboost__learning_rate": [0.1],#[0.01, 0.1, 1.0],
    },
]