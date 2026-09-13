import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_NAME = "data.csv"

df = pd.read_csv(
    FILE_NAME,
    sep=";"
)

# Remove spaces/tabs from column names
df.columns = df.columns.str.strip()


print("\n======================================")
print("DATASET LOADED")
print("======================================")

print("Dataset Shape:", df.shape)


# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()

print("Duplicate Rows:", duplicates)

if duplicates > 0:

    df = df.drop_duplicates().copy()

    print("Duplicates removed.")


# ============================================================
# 3. CHECK TARGET
# ============================================================

if "Target" not in df.columns:

    raise ValueError(
        "Target column was not found."
    )


print("\nTarget Distribution:")

print(
    df["Target"].value_counts()
)


# ============================================================
# 4. SELECT HUMAN-READABLE FEATURES
# ============================================================

base_features = [

    "Age at enrollment",

    "Gender",

    "Admission grade",

    "Scholarship holder",

    "Debtor",

    "Tuition fees up to date",

    "Curricular units 1st sem (enrolled)",

    "Curricular units 1st sem (approved)",

    "Curricular units 1st sem (grade)",

    "Curricular units 2nd sem (enrolled)",

    "Curricular units 2nd sem (approved)",

    "Curricular units 2nd sem (grade)"
]


# Check every selected feature exists
for feature in base_features:

    if feature not in df.columns:

        raise ValueError(
            f"Missing column: {feature}"
        )


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# Feature 1
# Semester 1 Pass Rate
# ------------------------------------------------------------

df["Semester1_Pass_Rate"] = np.where(

    df[
        "Curricular units 1st sem (enrolled)"
    ] > 0,

    df[
        "Curricular units 1st sem (approved)"
    ]
    /
    df[
        "Curricular units 1st sem (enrolled)"
    ],

    0
)


# ------------------------------------------------------------
# Feature 2
# Semester 2 Pass Rate
# ------------------------------------------------------------

df["Semester2_Pass_Rate"] = np.where(

    df[
        "Curricular units 2nd sem (enrolled)"
    ] > 0,

    df[
        "Curricular units 2nd sem (approved)"
    ]
    /
    df[
        "Curricular units 2nd sem (enrolled)"
    ],

    0
)


# ------------------------------------------------------------
# Feature 3
# Average Semester Grade
# ------------------------------------------------------------

df["Average_Semester_Grade"] = (

    df[
        "Curricular units 1st sem (grade)"
    ]

    +

    df[
        "Curricular units 2nd sem (grade)"
    ]

) / 2


# ------------------------------------------------------------
# Feature 4
# Academic Progress
# ------------------------------------------------------------

df["Academic_Progress"] = (

    df[
        "Curricular units 2nd sem (grade)"
    ]

    -

    df[
        "Curricular units 1st sem (grade)"
    ]
)


# ------------------------------------------------------------
# Feature 5
# Total Approved Units
# ------------------------------------------------------------

df["Total_Approved_Units"] = (

    df[
        "Curricular units 1st sem (approved)"
    ]

    +

    df[
        "Curricular units 2nd sem (approved)"
    ]
)


# ------------------------------------------------------------
# Feature 6
# Financial Risk
# ------------------------------------------------------------

df["Financial_Risk"] = (

    df["Debtor"]

    +

    (
        1
        -
        df["Tuition fees up to date"]
    )
)


# ------------------------------------------------------------
# Feature 7
# Age Group
# ------------------------------------------------------------

df["Age_Group"] = pd.cut(

    df["Age at enrollment"],

    bins=[
        0,
        20,
        25,
        30,
        40,
        100
    ],

    labels=[
        0,
        1,
        2,
        3,
        4
    ],

    include_lowest=True

).astype(int)


# ============================================================
# 6. ENGINEERED FEATURE LIST
# ============================================================

engineered_features = [

    "Semester1_Pass_Rate",

    "Semester2_Pass_Rate",

    "Average_Semester_Grade",

    "Academic_Progress",

    "Total_Approved_Units",

    "Financial_Risk",

    "Age_Group"
]


# ============================================================
# 7. FINAL MODEL FEATURES
# ============================================================

all_features = (
    base_features
    +
    engineered_features
)


X = df[
    all_features
].copy()


y = df[
    "Target"
].copy()


print("\n======================================")
print("MODEL FEATURES")
print("======================================")

for feature in all_features:

    print("-", feature)


print(
    "\nUser-entered features:",
    len(base_features)
)

print(
    "Engineered features:",
    len(engineered_features)
)

print(
    "Total model features:",
    X.shape[1]
)


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n======================================")
print("TRAIN / TEST SPLIT")
print("======================================")

print(
    "Training Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)


# ============================================================
# 9. CREATE MODELS
# ============================================================

models = {

    # Logistic Regression needs scaling
    "Logistic Regression":

        Pipeline([

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",

                LogisticRegression(

                    max_iter=5000,

                    random_state=42
                )
            )
        ]),


    "Decision Tree":

        DecisionTreeClassifier(

            random_state=42,

            class_weight="balanced"
        ),


    "Random Forest":

        RandomForestClassifier(

            n_estimators=300,

            random_state=42,

            class_weight="balanced",

            n_jobs=-1
        ),


    "Gradient Boosting":

        GradientBoostingClassifier(

            random_state=42
        )
}


# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}


for model_name, model in models.items():

    print("\n")
    print("=" * 60)
    print(model_name)
    print("=" * 60)


    # Train
    model.fit(
        X_train,
        y_train
    )


    # Predict
    predictions = model.predict(
        X_test
    )


    # Metrics

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(

        y_test,
        predictions,

        average="weighted",

        zero_division=0
    )


    recall = recall_score(

        y_test,
        predictions,

        average="weighted",

        zero_division=0
    )


    f1 = f1_score(

        y_test,
        predictions,

        average="weighted",

        zero_division=0
    )


    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )


    print(
        "\nClassification Report:"
    )

    print(

        classification_report(

            y_test,
            predictions,

            zero_division=0
        )
    )


    print(
        "Confusion Matrix:"
    )

    print(

        confusion_matrix(

            y_test,
            predictions
        )
    )


    results.append({

        "Model":
            model_name,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1 Score":
            f1
    })


    trained_models[
        model_name
    ] = model


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(

    by="F1 Score",

    ascending=False
)


print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(

    results_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)


best_model = trained_models[
    best_model_name
]


best_f1 = (
    results_df.iloc[0]["F1 Score"]
)


print("\n======================================")
print("BEST MODEL")
print("======================================")

print(
    "Model:",
    best_model_name
)

print(
    "Weighted F1 Score:",
    round(
        best_f1,
        4
    )
)


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

joblib.dump(

    best_model,

    "student_risk_model.pkl"
)


# Exact feature order
joblib.dump(

    all_features,

    "model_features.pkl"
)


# Model name
joblib.dump(

    best_model_name,

    "model_name.pkl"
)


print("\n======================================")
print("FILES SAVED SUCCESSFULLY")
print("======================================")

print(
    "student_risk_model.pkl"
)

print(
    "model_features.pkl"
)

print(
    "model_name.pkl"
)

print(
    "\nModel expects:",
    len(all_features),
    "features"
)