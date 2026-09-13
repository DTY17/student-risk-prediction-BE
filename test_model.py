import pandas as pd
import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "student_risk_model.pkl"
)

model_features = joblib.load(
    "model_features.pkl"
)

model_name = joblib.load(
    "model_name.pkl"
)


print(
    "Model loaded successfully"
)

print(
    "Model:",
    model_name
)

print(
    "Number of features:",
    len(model_features)
)


# ============================================================
# EXAMPLE STUDENT
# ============================================================

age = 21

gender = "Male"

admission_grade = 135.0

scholarship_holder = True

debtor = False

tuition_fees_up_to_date = True


semester1_enrolled_units = 6

semester1_approved_units = 5

semester1_grade = 13.5


semester2_enrolled_units = 6

semester2_approved_units = 5

semester2_grade = 14.0


# ============================================================
# CONVERT VALUES
# ============================================================

gender_value = (
    1
    if gender == "Male"
    else 0
)

scholarship_value = int(
    scholarship_holder
)

debtor_value = int(
    debtor
)

tuition_value = int(
    tuition_fees_up_to_date
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

semester1_pass_rate = (

    semester1_approved_units
    /
    semester1_enrolled_units

    if semester1_enrolled_units > 0

    else 0
)


semester2_pass_rate = (

    semester2_approved_units
    /
    semester2_enrolled_units

    if semester2_enrolled_units > 0

    else 0
)


average_grade = (

    semester1_grade

    +

    semester2_grade

) / 2


academic_progress = (

    semester2_grade

    -

    semester1_grade
)


total_approved = (

    semester1_approved_units

    +

    semester2_approved_units
)


financial_risk = (

    debtor_value

    +

    (
        1
        -
        tuition_value
    )
)


if age <= 20:

    age_group = 0

elif age <= 25:

    age_group = 1

elif age <= 30:

    age_group = 2

elif age <= 40:

    age_group = 3

else:

    age_group = 4


# ============================================================
# DATAFRAME
# ============================================================

student_data = {

    "Age at enrollment":
        age,

    "Gender":
        gender_value,

    "Admission grade":
        admission_grade,

    "Scholarship holder":
        scholarship_value,

    "Debtor":
        debtor_value,

    "Tuition fees up to date":
        tuition_value,

    "Curricular units 1st sem (enrolled)":
        semester1_enrolled_units,

    "Curricular units 1st sem (approved)":
        semester1_approved_units,

    "Curricular units 1st sem (grade)":
        semester1_grade,

    "Curricular units 2nd sem (enrolled)":
        semester2_enrolled_units,

    "Curricular units 2nd sem (approved)":
        semester2_approved_units,

    "Curricular units 2nd sem (grade)":
        semester2_grade,

    "Semester1_Pass_Rate":
        semester1_pass_rate,

    "Semester2_Pass_Rate":
        semester2_pass_rate,

    "Average_Semester_Grade":
        average_grade,

    "Academic_Progress":
        academic_progress,

    "Total_Approved_Units":
        total_approved,

    "Financial_Risk":
        financial_risk,

    "Age_Group":
        age_group
}


student = pd.DataFrame(
    [student_data]
)


student = student.reindex(
    columns=model_features
)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    student
)[0]


probabilities = model.predict_proba(
    student
)[0]


# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print("PREDICTION")
print("======================================")

print(
    "Prediction:",
    prediction
)


print(
    "\nProbabilities:"
)


for class_name, probability in zip(

    model.classes_,
    probabilities

):

    print(

        class_name,

        ":",

        round(
            probability * 100,
            2
        ),

        "%"
    )