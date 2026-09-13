from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from typing import Literal

import pandas as pd
import joblib
import os


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(

    title="Student Academic Risk Prediction API",

    description=(
        "Predict whether a student is likely to "
        "Dropout, remain Enrolled, or Graduate."
    ),

    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "http://localhost:5173",

        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(

    BASE_DIR,

    "student_risk_model.pkl"
)


FEATURE_PATH = os.path.join(

    BASE_DIR,

    "model_features.pkl"
)


MODEL_NAME_PATH = os.path.join(

    BASE_DIR,

    "model_name.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(
        MODEL_PATH
    )

    model_features = joblib.load(
        FEATURE_PATH
    )

    model_name = joblib.load(
        MODEL_NAME_PATH
    )


    print(
        "======================================"
    )

    print(
        "MODEL LOADED SUCCESSFULLY"
    )

    print(
        "======================================"
    )

    print(
        "Selected Model:",
        model_name
    )

    print(
        "Features Expected:",
        len(model_features)
    )


except Exception as error:

    print(
        "MODEL LOADING ERROR:",
        error
    )

    model = None

    model_features = None

    model_name = None


# ============================================================
# USER INPUT MODEL
# ============================================================

class StudentData(BaseModel):

    # Personal / Admission
    age: int = Field(
        ge=15,
        le=100
    )

    gender: Literal[
        "Male",
        "Female"
    ]

    admission_grade: float = Field(
        ge=0,
        le=200
    )


    # Financial
    scholarship_holder: bool

    debtor: bool

    tuition_fees_up_to_date: bool


    # Semester 1
    semester1_enrolled_units: int = Field(
        ge=0
    )

    semester1_approved_units: int = Field(
        ge=0
    )

    semester1_grade: float = Field(
        ge=0,
        le=20
    )


    # Semester 2
    semester2_enrolled_units: int = Field(
        ge=0
    )

    semester2_approved_units: int = Field(
        ge=0
    )

    semester2_grade: float = Field(
        ge=0,
        le=20
    )


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Student Academic Risk Prediction API",

        "version":
            "2.0",

        "status":
            "running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "OK",

        "model_loaded":
            model is not None,

        "model_name":
            model_name,

        "features_expected":
            (
                len(model_features)

                if model_features

                else 0
            )
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict_student(
    data: StudentData
):


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if model is None:

        raise HTTPException(

            status_code=500,

            detail="ML model is not loaded."
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    if (
        data.semester1_approved_units
        >
        data.semester1_enrolled_units
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "Semester 1 approved units "
                "cannot be greater than "
                "Semester 1 enrolled units."
            )
        )


    if (
        data.semester2_approved_units
        >
        data.semester2_enrolled_units
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "Semester 2 approved units "
                "cannot be greater than "
                "Semester 2 enrolled units."
            )
        )


    try:

        # ====================================================
        # CONVERT HUMAN-READABLE VALUES
        # ====================================================

        # Dataset:
        # Male   = 1
        # Female = 0

        gender_value = (

            1

            if data.gender == "Male"

            else 0
        )


        scholarship_value = int(
            data.scholarship_holder
        )


        debtor_value = int(
            data.debtor
        )


        tuition_value = int(
            data.tuition_fees_up_to_date
        )


        # ====================================================
        # USER-ENTERED FEATURES
        # ====================================================

        student_data = {

            "Age at enrollment":
                data.age,

            "Gender":
                gender_value,

            "Admission grade":
                data.admission_grade,

            "Scholarship holder":
                scholarship_value,

            "Debtor":
                debtor_value,

            "Tuition fees up to date":
                tuition_value,

            "Curricular units 1st sem (enrolled)":
                data.semester1_enrolled_units,

            "Curricular units 1st sem (approved)":
                data.semester1_approved_units,

            "Curricular units 1st sem (grade)":
                data.semester1_grade,

            "Curricular units 2nd sem (enrolled)":
                data.semester2_enrolled_units,

            "Curricular units 2nd sem (approved)":
                data.semester2_approved_units,

            "Curricular units 2nd sem (grade)":
                data.semester2_grade
        }


        # ====================================================
        # FEATURE 1
        # SEMESTER 1 PASS RATE
        # ====================================================

        if data.semester1_enrolled_units > 0:

            semester1_pass_rate = (

                data.semester1_approved_units

                /

                data.semester1_enrolled_units
            )

        else:

            semester1_pass_rate = 0


        student_data[
            "Semester1_Pass_Rate"
        ] = semester1_pass_rate


        # ====================================================
        # FEATURE 2
        # SEMESTER 2 PASS RATE
        # ====================================================

        if data.semester2_enrolled_units > 0:

            semester2_pass_rate = (

                data.semester2_approved_units

                /

                data.semester2_enrolled_units
            )

        else:

            semester2_pass_rate = 0


        student_data[
            "Semester2_Pass_Rate"
        ] = semester2_pass_rate


        # ====================================================
        # FEATURE 3
        # AVERAGE SEMESTER GRADE
        # ====================================================

        average_grade = (

            data.semester1_grade

            +

            data.semester2_grade

        ) / 2


        student_data[
            "Average_Semester_Grade"
        ] = average_grade


        # ====================================================
        # FEATURE 4
        # ACADEMIC PROGRESS
        # ====================================================

        academic_progress = (

            data.semester2_grade

            -

            data.semester1_grade
        )


        student_data[
            "Academic_Progress"
        ] = academic_progress


        # ====================================================
        # FEATURE 5
        # TOTAL APPROVED UNITS
        # ====================================================

        total_approved = (

            data.semester1_approved_units

            +

            data.semester2_approved_units
        )


        student_data[
            "Total_Approved_Units"
        ] = total_approved


        # ====================================================
        # FEATURE 6
        # FINANCIAL RISK
        # ====================================================

        financial_risk = (

            debtor_value

            +

            (
                1
                -
                tuition_value
            )
        )


        student_data[
            "Financial_Risk"
        ] = financial_risk


        # ====================================================
        # FEATURE 7
        # AGE GROUP
        # ====================================================

        if data.age <= 20:

            age_group = 0


        elif data.age <= 25:

            age_group = 1


        elif data.age <= 30:

            age_group = 2


        elif data.age <= 40:

            age_group = 3


        else:

            age_group = 4


        student_data[
            "Age_Group"
        ] = age_group


        # ====================================================
        # CREATE DATAFRAME
        # ====================================================

        student_df = pd.DataFrame(
            [student_data]
        )


        # Force exact feature order used during training
        student_df = student_df.reindex(
            columns=model_features
        )


        # ====================================================
        # CHECK FEATURES
        # ====================================================

        if student_df.isnull().any().any():

            missing_features = (

                student_df.columns[

                    student_df.isnull().any()

                ].tolist()
            )


            raise ValueError(

                "Missing model features: "

                +

                str(missing_features)
            )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(
            student_df
        )[0]


        # ====================================================
        # PROBABILITIES
        # ====================================================

        probabilities = model.predict_proba(
            student_df
        )[0]


        probability_result = {}


        for class_name, probability in zip(

            model.classes_,

            probabilities

        ):

            probability_result[
                str(class_name)
            ] = round(

                float(probability)
                *
                100,

                2
            )


        # ====================================================
        # RISK LEVEL
        # ====================================================

        if prediction == "Dropout":

            risk_level = "HIGH"


        elif prediction == "Enrolled":

            risk_level = "MEDIUM"


        else:

            risk_level = "LOW"


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "success":
                True,

            "prediction":
                str(prediction),

            "risk_level":
                risk_level,

            "probabilities":
                probability_result,

            "student_summary": {

                "age":
                    data.age,

                "gender":
                    data.gender,

                "admission_grade":
                    data.admission_grade,

                "semester1_pass_rate":
                    round(
                        semester1_pass_rate * 100,
                        2
                    ),

                "semester2_pass_rate":
                    round(
                        semester2_pass_rate * 100,
                        2
                    ),

                "average_semester_grade":
                    round(
                        average_grade,
                        2
                    ),

                "academic_progress":
                    round(
                        academic_progress,
                        2
                    ),

                "total_approved_units":
                    total_approved,

                "financial_risk_score":
                    financial_risk
            }
        }


    except HTTPException:

        raise


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )