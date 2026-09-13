# Student Academic Risk Prediction System - Backend

This is the Python backend for the **Student Academic Risk Prediction System**.

The backend is built using:

- Python
- FastAPI
- SQLAlchemy
- MySQL
- PyMySQL
- Pydantic
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Uvicorn

The backend handles student CRUD operations, database access, feature engineering, and Machine Learning prediction.

---

## Main Features

The backend supports:

- Create student records
- Retrieve all students
- Retrieve a single student
- Update student records
- Delete student records
- Connect to MySQL
- Load the trained Machine Learning model
- Perform feature engineering
- Generate academic outcome predictions
- Generate prediction probabilities
- Return academic risk levels
- Provide Swagger API documentation

---

## Prediction Classes

The trained Machine Learning model predicts one of three classes:

- Dropout
- Enrolled
- Graduate

The application interprets these predictions as:

| Prediction | Risk Level |
|---|---|
| Dropout | HIGH |
| Enrolled | MEDIUM |
| Graduate | LOW |

---

## Technologies Used

- Python
- FastAPI
- SQLAlchemy
- MySQL
- PyMySQL
- Pydantic
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Uvicorn
- python-dotenv

---

## Project Structure

```text
backend/
│
├── api.py
├── database.py
├── models.py
├── schemas.py
├── prediction_service.py
│
├── train_model.py
├── test_model.py
├── data.csv
│
├── student_risk_model.pkl
├── model_features.pkl
├── model_name.pkl
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## File Description

### `api.py`

The main FastAPI application.

Responsibilities:

- Create the FastAPI application
- Configure CORS
- Provide CRUD endpoints
- Provide prediction endpoints
- Connect request validation with database operations
- Call the Machine Learning prediction service

Main endpoints:

```text
GET     /
GET     /health
POST    /students
GET     /students
GET     /students/{student_id}
PUT     /students/{student_id}
DELETE  /students/{student_id}
POST    /predict
POST    /students/{student_id}/predict
```

### `database.py`

Handles the MySQL connection using SQLAlchemy.

Responsibilities:

- Load database environment variables
- Create the SQLAlchemy database URL
- Create the database engine
- Create `SessionLocal`
- Create `Base`
- Provide the `get_db()` dependency

### `models.py`

Contains SQLAlchemy database models.

Main table:

```text
students
```

The student table contains:

- id
- name
- age
- gender
- admission_grade
- scholarship_holder
- debtor
- tuition_fees_up_to_date
- semester1_enrolled_units
- semester1_approved_units
- semester1_grade
- semester2_enrolled_units
- semester2_approved_units
- semester2_grade

### `schemas.py`

Contains Pydantic models for input validation.

Main schemas:

- StudentData
- StudentCreate
- StudentUpdate

### `prediction_service.py`

Contains the Machine Learning prediction logic.

Responsibilities:

- Load the trained model
- Load the feature list
- Load the selected model name
- Convert frontend values into model values
- Perform feature engineering
- Create the model DataFrame
- Match feature order with training
- Generate prediction
- Generate probabilities
- Calculate risk level
- Return student summary

### `train_model.py`

Used to:

- Load the dataset
- Clean the dataset
- Select model features
- Perform feature engineering
- Split training and testing data
- Train multiple Machine Learning algorithms
- Evaluate the models
- Select the best model
- Save the trained model
- Save the model feature list
- Save the selected model name

### `test_model.py`

Used to test the saved Machine Learning model before API integration.

It verifies:

- Model loading
- Feature loading
- Input transformation
- Feature engineering
- Prediction
- Prediction probabilities

### `data.csv`

Contains the student academic dataset used for model training.

### `student_risk_model.pkl`

Contains the trained Machine Learning model selected during training.

### `model_features.pkl`

Contains the exact feature names and order expected by the trained model.

### `model_name.pkl`

Contains the name of the selected Machine Learning algorithm.

---

## Machine Learning Input

The application asks the user for 12 student fields:

1. Age
2. Gender
3. Admission Grade
4. Scholarship Holder
5. Debtor
6. Tuition Fees Up To Date
7. Semester 1 Enrolled Units
8. Semester 1 Approved Units
9. Semester 1 Average Grade
10. Semester 2 Enrolled Units
11. Semester 2 Approved Units
12. Semester 2 Average Grade

---

## Feature Engineering

The backend automatically generates 7 additional features.

### 1. Semester 1 Pass Rate

```text
Semester1_Pass_Rate = Semester 1 Approved Units / Semester 1 Enrolled Units
```

### 2. Semester 2 Pass Rate

```text
Semester2_Pass_Rate = Semester 2 Approved Units / Semester 2 Enrolled Units
```

### 3. Average Semester Grade

```text
Average_Semester_Grade = (Semester 1 Grade + Semester 2 Grade) / 2
```

### 4. Academic Progress

```text
Academic_Progress = Semester 2 Grade - Semester 1 Grade
```

### 5. Total Approved Units

```text
Total_Approved_Units = Semester 1 Approved Units + Semester 2 Approved Units
```

### 6. Financial Risk

```text
Financial_Risk = Debtor + (1 - Tuition Fees Up To Date)
```

### 7. Age Group

Age is converted into a grouped numerical feature before prediction.

---

## Total Model Features

```text
12 user input features
+
7 engineered features
=
19 total model features
```

---

## Requirements

Install Python first.

Check the installed version:

```bash
python --version
```

---

## Create a Virtual Environment

Recommended:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

---

## Install Dependencies

Run:

```bash
python -m pip install -r requirements.txt
```

Example `requirements.txt`:

```text
fastapi
uvicorn
pydantic
pandas
numpy
scikit-learn
joblib
sqlalchemy
pymysql
python-dotenv
```

---

## MySQL Setup

Create the database:

```sql
CREATE DATABASE student_risk_prediction;
```

The students table can be created automatically by SQLAlchemy when the application starts.

---

## Environment Variables

Create a file named:

```text
.env
```

Example:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=YOUR_MYSQL_PASSWORD
DB_NAME=student_risk_prediction
```

Do not upload `.env` to GitHub.

---

## `.gitignore`

Example:

```gitignore
.env
__pycache__/
*.pyc
venv/
.venv/
```

---

## Train the Machine Learning Model

Run:

```bash
python train_model.py
```

The training process should generate:

```text
student_risk_model.pkl
model_features.pkl
model_name.pkl
```

---

## Test the Trained Model

Run:

```bash
python test_model.py
```

---

## Run the Backend

Start FastAPI:

```bash
python -m uvicorn api:app --reload
```

The API normally runs at:

```text
http://127.0.0.1:8000
```

---

## Swagger API Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Home

```text
GET /
```

### Health Check

```text
GET /health
```

Checks:

- Backend status
- Model loading status
- Selected model name
- Number of expected model features

### Create Student

```text
POST /students
```

Example request:

```json
{
  "name": "John Silva",
  "age": 21,
  "gender": "Male",
  "admission_grade": 135,
  "scholarship_holder": true,
  "debtor": false,
  "tuition_fees_up_to_date": true,
  "semester1_enrolled_units": 6,
  "semester1_approved_units": 5,
  "semester1_grade": 13.5,
  "semester2_enrolled_units": 6,
  "semester2_approved_units": 5,
  "semester2_grade": 14
}
```

### Get All Students

```text
GET /students
```

### Get One Student

```text
GET /students/{student_id}
```

### Update Student

```text
PUT /students/{student_id}
```

### Delete Student

```text
DELETE /students/{student_id}
```

### Direct Prediction

```text
POST /predict
```

This endpoint accepts student information directly without first saving it in MySQL.

### Predict Saved Student

```text
POST /students/{student_id}/predict
```

Prediction flow:

```text
Student ID
   ↓
FastAPI
   ↓
MySQL
   ↓
Load Student
   ↓
Feature Engineering
   ↓
Machine Learning Model
   ↓
Prediction
```

---

## Example Prediction Response

```json
{
  "student": {
    "id": 1,
    "name": "John Silva"
  },
  "success": true,
  "prediction": "Graduate",
  "risk_level": "LOW",
  "probabilities": {
    "Dropout": 8.2,
    "Enrolled": 14.5,
    "Graduate": 77.3
  },
  "student_summary": {
    "age": 21,
    "gender": "Male",
    "admission_grade": 135,
    "semester1_pass_rate": 83.33,
    "semester2_pass_rate": 83.33,
    "average_semester_grade": 13.75,
    "academic_progress": 0.5,
    "total_approved_units": 10,
    "financial_risk_score": 0
  }
}
```

The probability values above are examples only. Actual values depend on the trained model.

---

## Backend Architecture

```text
                    FastAPI
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
     SQLAlchemy              Prediction Service
          │                         │
          ▼                         ▼
        MySQL                  ML Model Files
          │                         │
          ▼                         ▼
   Student Records        Dropout / Enrolled / Graduate
```

---

## Complete Application Architecture

```text
React + TypeScript
       │
       ▼
    FastAPI
       │
   ┌───┴─────────────┐
   │                 │
   ▼                 ▼
 MySQL            ML Model
   │                 │
   ▼                 ▼
Student Data      Prediction
                     │
                     ▼
          Dropout / Enrolled / Graduate
```

---

## Validation

The backend validates conditions such as:

- Age must be within the accepted range
- Admission Grade must be valid
- Semester grades must be within the accepted range
- Semester 1 approved units cannot exceed Semester 1 enrolled units
- Semester 2 approved units cannot exceed Semester 2 enrolled units

---

## CORS

During local development, the backend allows requests from:

```text
http://localhost:5173
http://127.0.0.1:5173
```

---

## Security Notes

- Never commit `.env`
- Never expose MySQL passwords in source code
- Keep production database credentials secure
- Validate all user input
- Restrict CORS before production deployment
- Do not hard-code sensitive credentials

---

## Important Machine Learning Note

This system uses both Semester 1 and Semester 2 academic information.

Therefore, it should be described as an:

**Academic Risk and Performance Prediction System using semester performance data**

rather than claiming that it predicts dropout immediately when the student first enrolls.

---

## Purpose of the System

The purpose of this project is to demonstrate the complete Machine Learning development and integration process.

The system combines:

- Dataset analysis
- Data preprocessing
- Feature engineering
- Machine Learning model training
- Model evaluation
- Model serialization
- REST API development
- MySQL database integration
- React frontend integration
- Real-time prediction output
