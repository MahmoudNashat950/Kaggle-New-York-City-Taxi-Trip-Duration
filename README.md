NYC Taxi Trip Duration Prediction
Overview

Accurately predicting taxi trip duration is an important task for improving transportation services, optimizing routes, and enhancing customer experience. In this project, we develop a machine learning model to estimate the duration of taxi trips in New York City using historical trip data and advanced feature engineering techniques.

The project combines data exploration, geographical calculations, time-based feature extraction, and Ridge Regression to build a robust and reproducible prediction pipeline. The entire workflow follows professional machine learning practices, from data preprocessing to model evaluation and deployment-ready model persistence.

Dataset Description

The dataset contains historical New York City taxi trip records, including pickup and drop-off information, timestamps, and location coordinates.

Key data attributes include:

Pickup datetime
Drop-off datetime
Pickup latitude and longitude
Drop-off latitude and longitude
Passenger count
Trip duration (target variable)

The objective is to predict the trip duration based on the available trip information.

Project Objectives

This project aims to:

Explore and understand taxi trip patterns in New York City.
Engineer meaningful geographical and temporal features.
Build a predictive machine learning model for trip duration estimation.
Evaluate model performance using regression metrics.
Create a reproducible and professional machine learning pipeline.
Save and load trained models for future predictions.
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Jupyter Notebook
Project Workflow
1. Exploratory Data Analysis (EDA)

The project begins with a detailed exploration of the dataset to understand:

Data distributions
Missing values
Outliers
Relationships between features
Geographic patterns of taxi trips
Trip duration trends over time

Visualizations are used extensively to gain insights and support feature selection.

2. Feature Engineering

Feature engineering plays a crucial role in improving prediction accuracy.

Implemented features include:

Distance-Based Features
Haversine distance between pickup and drop-off locations
Manhattan distance approximation
Direction/Bearing calculations
Time-Based Features
Hour of the day
Day of the week
Month
Weekend indicator
Rush-hour identification
Polynomial Features

Polynomial transformations are applied to capture nonlinear relationships between variables and trip duration.

3. Data Preprocessing

The preprocessing pipeline includes:

Handling missing values
Removing abnormal records and outliers
Feature scaling
Train-test splitting
Feature transformation and encoding

The workflow is designed to be fully reproducible.

4. Model Training

The primary model used in this project is:

Ridge Regression

Ridge Regression helps reduce overfitting by applying L2 regularization while maintaining strong predictive performance.

Model training includes:

Hyperparameter tuning
Cross-validation
Regularized regression fitting
5. Model Evaluation

The trained model is evaluated using common regression metrics such as:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R² Score

These metrics provide a comprehensive assessment of prediction accuracy and model generalization.

6. Model Persistence

To support deployment and future inference, the project includes:

Saving trained models using Joblib
Loading saved models for predictions
Reusable prediction workflows

This allows the model to be integrated into production systems without retraining.

Results

The feature engineering process significantly improves model performance by incorporating both geographical and temporal information.

Key findings include:

Trip distance is one of the strongest predictors of travel time.
Time-of-day features capture traffic-related patterns.
Polynomial features help model nonlinear relationships.
Ridge Regression provides a good balance between bias and variance.
Project Structure
├── data/
├── notebooks/
├── models/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── saved_models/
├── requirements.txt
└── README.md
Future Improvements

Potential enhancements include:

Implementing advanced ensemble models such as Random Forest and XGBoost.
Incorporating real-time traffic and weather information.
Applying deep learning techniques for improved prediction accuracy.
Deploying the model as a web application or REST API.
Integrating geospatial visualization dashboards.
Conclusion

This project demonstrates how machine learning can be used to accurately predict New York City taxi trip durations through effective data analysis and feature engineering. By leveraging distance calculations, temporal features, polynomial transformations, and Ridge Regression, the model provides reliable travel time estimates while maintaining a clean, reproducible, and production-ready workflow.

Author

Mahmoud Nashaat
Computer & Artificial Intelligence Engineering Student
Machine Learning & Data Science Enthusiast
Full-Stack .NET Developer
