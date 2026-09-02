import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/StudentPerformanceFactors.csv")

print("Dataset Shape:", df.shape)


# --------------------------------------------------
# 2. DATA VALIDATION
# --------------------------------------------------

df = df[df["Exam_Score"].between(0, 100)].copy()

print("Shape after target validation:", df.shape)


# --------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]


# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 5. IDENTIFY FEATURE TYPES
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()


# --------------------------------------------------
# 6. NUMERICAL PREPROCESSING
# --------------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# --------------------------------------------------
# 7. CATEGORICAL PREPROCESSING
# --------------------------------------------------

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# --------------------------------------------------
# 8. COMBINE PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# --------------------------------------------------
# 9. DEFINE MODELS
# --------------------------------------------------

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


# --------------------------------------------------
# 10. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

results = {}
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Store results
    results[name] = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    trained_models[name] = pipeline

    print(f"\n{name}")
    print("-------------------------")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R²  :", r2)


# --------------------------------------------------
# 11. SELECT BEST MODEL
# --------------------------------------------------

best_model_name = max(
    results,
    key=lambda name: results[name]["R2"]
)

best_model = trained_models[best_model_name]

print("\n==============================")
print("BEST MODEL:", best_model_name)
print("==============================")


# --------------------------------------------------
# 12. SAVE MODEL METRICS
# --------------------------------------------------

with open("models/metrics.json", "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )

print("\nMetrics saved to: models/metrics.json")


# --------------------------------------------------
# 13. SAVE / DUMP BEST MODEL
# --------------------------------------------------

joblib.dump(
    best_model,
    "models/student_performance_model.pkl"
)

print("Model dumped successfully.")
print("Saved to: models/student_performance_model.pkl")