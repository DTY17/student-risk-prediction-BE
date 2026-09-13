from fastapi import (
    Depends,
    FastAPI,
    HTTPException
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from sqlalchemy.orm import Session


from database import (
    Base,
    engine,
    get_db
)

from models import StudentDB

from schemas import (
    StudentCreate,
    StudentData,
    StudentUpdate
)

from prediction_service import (
    model,
    model_features,
    model_name,
    run_prediction
)


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

Base.metadata.create_all(
    bind=engine
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(

    title=(
        "Student Academic Risk "
        "Prediction API"
    ),

    description=(
        "Student CRUD operations and "
        "academic risk prediction."
    ),

    version="3.0.0"
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
# HELPER
# ============================================================

def student_to_dict(
    student: StudentDB
):

    return {

        "id":
            student.id,

        "name":
            student.name,

        "age":
            student.age,

        "gender":
            student.gender,

        "admission_grade":
            student.admission_grade,

        "scholarship_holder":
            student.scholarship_holder,

        "debtor":
            student.debtor,

        "tuition_fees_up_to_date":
            student.tuition_fees_up_to_date,

        "semester1_enrolled_units":
            student.semester1_enrolled_units,

        "semester1_approved_units":
            student.semester1_approved_units,

        "semester1_grade":
            student.semester1_grade,

        "semester2_enrolled_units":
            student.semester2_enrolled_units,

        "semester2_approved_units":
            student.semester2_approved_units,

        "semester2_grade":
            student.semester2_grade
    }


def validate_student(
    data: StudentData
):

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
                "enrolled units."
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
                "enrolled units."
            )
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
            "3.0",

        "status":
            "running"
    }


# ============================================================
# HEALTH
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
# CREATE STUDENT
# ============================================================

@app.post("/students")
def create_student(

    data: StudentCreate,

    db: Session = Depends(
        get_db
    )
):

    validate_student(
        data
    )


    student = StudentDB(

        name=
            data.name,

        age=
            data.age,

        gender=
            data.gender,

        admission_grade=
            data.admission_grade,

        scholarship_holder=
            data.scholarship_holder,

        debtor=
            data.debtor,

        tuition_fees_up_to_date=
            data.tuition_fees_up_to_date,

        semester1_enrolled_units=
            data.semester1_enrolled_units,

        semester1_approved_units=
            data.semester1_approved_units,

        semester1_grade=
            data.semester1_grade,

        semester2_enrolled_units=
            data.semester2_enrolled_units,

        semester2_approved_units=
            data.semester2_approved_units,

        semester2_grade=
            data.semester2_grade
    )


    db.add(
        student
    )

    db.commit()

    db.refresh(
        student
    )


    return {

        "success":
            True,

        "message":
            "Student created successfully.",

        "student":
            student_to_dict(
                student
            )
    }


# ============================================================
# GET ALL STUDENTS
# ============================================================

@app.get("/students")
def get_students(

    db: Session = Depends(
        get_db
    )
):

    students = (

        db.query(
            StudentDB
        )

        .order_by(
            StudentDB.id.desc()
        )

        .all()
    )


    return {

        "success":
            True,

        "count":
            len(students),

        "students": [

            student_to_dict(
                student
            )

            for student in students

        ]
    }


# ============================================================
# GET ONE STUDENT
# ============================================================

@app.get(
    "/students/{student_id}"
)
def get_student(

    student_id: int,

    db: Session = Depends(
        get_db
    )
):

    student = (

        db.query(
            StudentDB
        )

        .filter(
            StudentDB.id
            ==
            student_id
        )

        .first()
    )


    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found."
        )


    return {

        "success":
            True,

        "student":
            student_to_dict(
                student
            )
    }


# ============================================================
# UPDATE STUDENT
# ============================================================

@app.put(
    "/students/{student_id}"
)
def update_student(

    student_id: int,

    data: StudentUpdate,

    db: Session = Depends(
        get_db
    )
):

    student = (

        db.query(
            StudentDB
        )

        .filter(
            StudentDB.id
            ==
            student_id
        )

        .first()
    )


    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found."
        )


    validate_student(
        data
    )


    student.name = (
        data.name
    )

    student.age = (
        data.age
    )

    student.gender = (
        data.gender
    )

    student.admission_grade = (
        data.admission_grade
    )

    student.scholarship_holder = (
        data.scholarship_holder
    )

    student.debtor = (
        data.debtor
    )

    student.tuition_fees_up_to_date = (
        data.tuition_fees_up_to_date
    )

    student.semester1_enrolled_units = (
        data.semester1_enrolled_units
    )

    student.semester1_approved_units = (
        data.semester1_approved_units
    )

    student.semester1_grade = (
        data.semester1_grade
    )

    student.semester2_enrolled_units = (
        data.semester2_enrolled_units
    )

    student.semester2_approved_units = (
        data.semester2_approved_units
    )

    student.semester2_grade = (
        data.semester2_grade
    )


    db.commit()

    db.refresh(
        student
    )


    return {

        "success":
            True,

        "message":
            "Student updated successfully.",

        "student":
            student_to_dict(
                student
            )
    }


# ============================================================
# DELETE STUDENT
# ============================================================

@app.delete(
    "/students/{student_id}"
)
def delete_student(

    student_id: int,

    db: Session = Depends(
        get_db
    )
):

    student = (

        db.query(
            StudentDB
        )

        .filter(
            StudentDB.id
            ==
            student_id
        )

        .first()
    )


    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found."
        )


    db.delete(
        student
    )

    db.commit()


    return {

        "success":
            True,

        "message":
            "Student deleted successfully."
    }


# ============================================================
# DIRECT PREDICTION
# ============================================================

@app.post("/predict")
def predict_student(
    data: StudentData
):

    try:

        return run_prediction(
            data
        )


    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# PREDICT SAVED STUDENT
# ============================================================

@app.post(
    "/students/{student_id}/predict"
)
def predict_saved_student(

    student_id: int,

    db: Session = Depends(
        get_db
    )
):

    student = (

        db.query(
            StudentDB
        )

        .filter(
            StudentDB.id
            ==
            student_id
        )

        .first()
    )


    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found."
        )


    data = StudentData(

        age=
            student.age,

        gender=
            student.gender,

        admission_grade=
            student.admission_grade,

        scholarship_holder=
            student.scholarship_holder,

        debtor=
            student.debtor,

        tuition_fees_up_to_date=
            student.tuition_fees_up_to_date,

        semester1_enrolled_units=
            student.semester1_enrolled_units,

        semester1_approved_units=
            student.semester1_approved_units,

        semester1_grade=
            student.semester1_grade,

        semester2_enrolled_units=
            student.semester2_enrolled_units,

        semester2_approved_units=
            student.semester2_approved_units,

        semester2_grade=
            student.semester2_grade
    )


    try:

        prediction = run_prediction(
            data
        )

        return {

            "student": {

                "id":
                    student.id,

                "name":
                    student.name
            },

            **prediction
        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )