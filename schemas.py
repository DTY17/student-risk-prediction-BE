from typing import Literal

from pydantic import BaseModel, Field


class StudentData(BaseModel):

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

    scholarship_holder: bool

    debtor: bool

    tuition_fees_up_to_date: bool

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


class StudentCreate(StudentData):

    name: str = Field(
        min_length=1,
        max_length=150
    )


class StudentUpdate(StudentCreate):

    pass