# Customer Booking Prediction

This project uses machine learning to predict whether a customer will complete a flight booking based on travel intent data such as booking origin, trip type, and lead time.

---

## Project Overview

**Objective:**  
To build a predictive model that determines if a customer will complete a booking (`booking_complete`) using structured travel-related features.

**Dataset:**  
Simulated version of `customer_booking.csv` containing features like:
- `booking_origin`, `trip_type`, `purchase_lead`
- `flight_duration`, `length_of_stay`, `sales_channel`
- `wants_extra_baggage`, `wants_preferred_seat`, etc.

---

## Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn (RandomForest, Pipeline, OneHotEncoder, metrics)
- imbalanced-learn (SMOTE)
- Matplotlib for visualizations

---

## Machine Learning Workflow

1. **Data Cleaning**
   - Checked missing values
   - Converted categorical features using `OneHotEncoder`

2. **Feature Engineering**
   - Focused on time-based features (`purchase_lead`, `flight_duration`)
   - Created interpretable dummy variables

3. **Handling Class Imbalance**
   - Applied **SMOTE** to oversample the minority class (`booking_complete = 1`)

4. **Model Training**
   - Used **RandomForestClassifier**
   - Pipeline: Preprocessing → SMOTE → Model

5. **Evaluation**
   - 5-fold Stratified Cross-Validation
   - Metrics reported: Accuracy, Precision, Recall, F1-score

---

## Cross-Validation Results

| Metric     | Score   |
|------------|---------|
| Accuracy   | 84.7%   |
| Precision  | 46.2%   |
| Recall     | 12.5%   |
| F1-score   | 19.7%   |

> The model performs well on predicting non-bookers, but struggles to catch the minority booking class — a typical challenge in imbalanced datasets.

---

## Feature Importance

Top 5 most important features influencing booking predictions:

1. `booking_origin_Australia`  
2. `booking_origin_Malaysia`  
3. `purchase_lead`  
4. `length_of_stay`  
5. `flight_duration`

> Booking origin and trip details play a crucial role in predicting customer booking behavior.

---

## Project Files

- `/notebooks/model_training.ipynb` — Full training pipeline and evaluation
- `/images/feature_importance.png` — Model interpretation chart
- `/presentation/final_slide.pptx` — Summary slide for business use
- `/data/sample_customer_booking.csv` — Sample dataset (non-sensitive)

---

## What I Learned

- Building interpretable machine learning pipelines using Scikit-learn
- Handling imbalanced data with SMOTE
- Evaluating models using cross-validation and proper metrics
- Translating technical insights into business decisions



