Churn Modelling — Predicting Customer Retention

This project focuses on predicting whether a customer will churn (leave) or stay with a company using machine learning and deep learning models.
It’s built around a classic business problem — understanding customer behavior — and includes everything from data preprocessing to model deployment with Streamlit.

📘 Overview

Customer churn prediction is one of the most useful ML applications in banking, telecom, and SaaS industries.
In this project, I trained and compared different models to predict churn using the Churn_Modelling.csv dataset.
The project also includes hyperparameter tuning, scaling, and a Streamlit app for easy model interaction.

🧠 What This Project Does

Loads and cleans customer data

Encodes categorical features (like Geography and Gender)

Scales numerical features for model performance

Trains multiple models including:

Artificial Neural Network (ANN)

Regression-based models

Tunes hyperparameters for optimal results

Saves trained models and encoders for reuse

Provides a Streamlit interface to test predictions in real time

🧰 Tech Stack

Python 3.x

TensorFlow / Keras – for ANN model

Scikit-learn – preprocessing, encoding, and model evaluation

Pandas, NumPy – data handling

Streamlit – web app for user interaction

Matplotlib / Seaborn – data visualization

📂 Project Structure
Churn-Modelling/
│
├── Churn_Modelling.csv              # Dataset

├── app.py                           # Main app (for running predictions)

├── streamlit_regression.py          # Streamlit web app

│


├── experiments.ipynb                # Data analysis & model experiments

├── hyperparametertuningann.ipynb    # ANN hyperparameter tuning

├── salaryregression.ipynb           # Regression comparison (for salary prediction)

├── prediction.ipynb                 # Testing predictions on sample inputs

│

├── model.h5                         # Trained ANN model

├── regression_model.h5              # Regression model

├── scaler.pkl                       # StandardScaler used for normalization

├── label_encoder_gender.pkl         # Encoder for gender feature

├── onehot_encoder_geo.pkl           # Encoder for geography feature

│

├── regressionlogs/fit/20251018-090651    # Training logs

├── requirements.txt                 # Python dependencies

└── README.md                         # (This file)

⚙️ How to Run the Project

Clone the repository

git clone https://github.com/Suraj3155/Churn-Modelling.git

cd Churn-Modelling


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run streamlit_regression.py


or, if you want to use the main script:

python app.py


Test predictions

Enter values for customer details like credit score, geography, gender, etc.

The app will predict if the customer is likely to leave (churn) or stay.

🔍 How It Works

The dataset (Churn_Modelling.csv) is first cleaned and processed.

Categorical data (like “France”, “Male”, “Female”) is encoded using pickled encoders (.pkl files).

Features are scaled using a saved scaler.pkl.

The ANN model (saved as model.h5) predicts churn probability.

Streamlit provides a simple web UI for input and prediction.

📈 Model Training

Used Keras Sequential API to build the ANN.

Performed hyperparameter tuning on:

Number of neurons

Activation functions

Optimizers (Adam, RMSprop, etc.)

Learning rate

Compared ANN results with regression-based baselines.

Best performing model saved as regression_model.h5.

🧩 Example Prediction
Feature	Value
Credit Score	720
Geography	France
Gender	Male
Age	35
Balance	10000
Estimated Salary	50000
Tenure	5
Has Credit Card	Yes
Is Active Member	Yes

Prediction: 🟢 Customer is likely to stay.

🚀 Future Improvements

Integrate advanced models (XGBoost, LightGBM, CatBoost)

Use deep neural networks with dropout and batch normalization

Add visual dashboards for churn analytics

Deploy on cloud (e.g., AWS or Streamlit Cloud)

🤝 Acknowledgements

Thanks to:

The open-source ML community

Tutorials and blogs that inspired the ANN model tuning

Kaggle datasets (especially the Churn Modelling dataset)

🪪 License

This project is open-source and free to use for learning or demonstration.
MIT License © 2025 Suraj Shivankar
