"""Model zoo. Class imbalance is handled with class weights inside each model;
SMOTE is offered only *inside* an imblearn pipeline so it never sees test folds."""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42


def make_models(y, include_smote=True):
    neg_pos = float((y == 0).sum()) / max(int((y == 1).sum()), 1)
    models = {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", C=0.1, max_iter=5000)),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                           class_weight="balanced_subsample", n_jobs=-1,
                                           random_state=RANDOM_STATE)),
        ]),
        "XGBoost": make_xgb(neg_pos),
    }
    if include_smote:
        models["Random Forest + SMOTE (train folds only)"] = ImbPipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                           n_jobs=-1, random_state=RANDOM_STATE)),
        ])
    return models


def make_xgb(neg_pos=1.0):
    # Shallow, strongly regularised trees: with only a few hundred injuries, depth 2
    # generalised better to unseen athletes than deeper trees.
    return XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=2, min_child_weight=20,
                         subsample=0.7, colsample_bytree=0.5, reg_lambda=10.0,
                         scale_pos_weight=neg_pos, eval_metric="aucpr", n_jobs=-1,
                         random_state=RANDOM_STATE)
