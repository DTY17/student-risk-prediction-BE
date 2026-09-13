from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    String
)

from database import Base


class StudentDB(Base):

    __tablename__ = "students"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True
    )

    name = Column(
        String(150),
        nullable=False
    )

    age = Column(
        Integer,
        nullable=False
    )

    gender = Column(
        String(20),
        nullable=False
    )

    admission_grade = Column(
        Float,
        nullable=False
    )

    scholarship_holder = Column(
        Boolean,
        nullable=False,
        default=False
    )

    debtor = Column(
        Boolean,
        nullable=False,
        default=False
    )

    tuition_fees_up_to_date = Column(
        Boolean,
        nullable=False,
        default=True
    )

    semester1_enrolled_units = Column(
        Integer,
        nullable=False
    )

    semester1_approved_units = Column(
        Integer,
        nullable=False
    )

    semester1_grade = Column(
        Float,
        nullable=False
    )

    semester2_enrolled_units = Column(
        Integer,
        nullable=False
    )

    semester2_approved_units = Column(
        Integer,
        nullable=False
    )

    semester2_grade = Column(
        Float,
        nullable=False
    )