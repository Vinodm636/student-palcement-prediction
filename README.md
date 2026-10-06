# Student Placement Prediction System

A machine learning web application built with Flask that predicts student placement outcomes using 8 classifiers with full preprocessing, feature engineering, and analytics.

---

## Features

- 8 ML models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, SVM, KNN, Naive Bayes
- Feature engineering with 7 composite scores
- Interactive analytics dashboard
- Skill-based recommendations for students
- Role-based login (admin / student)

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Vinodm636/student-palcement-prediction.git
cd student-palcement-prediction
```

### 2. Install Python dependencies

Make sure you have **Python 3.8+** installed, then run:

```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the application

```bash
python app.py
```

### 4. Open in browser

Go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Login Credentials

| Role    | Username | Password |
|---------|----------|----------|
| Admin   | admin    | admin123 |
| Student | student  | student123 |

> Check `users.py` for all available accounts.

---

## Project Structure

```
Student_Placement_Prediction/
├── app.py                  # Main Flask application
├── generate_data.py        # Dataset generation script
├── placement_data.csv      # Student dataset
├── users.py                # User credentials
├── verify.py               # Verification utilities
├── templates/              # HTML templates
│   ├── login.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── analytics.html
│   └── ...
└── static/charts/          # Generated chart images
```

---

## Requirements

- Python 3.8+
- flask
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
