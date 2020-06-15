# Disaster-Response-ML-Project
This project involves ETL and a Machine Learning Pipeline implementation that classifies tweets to assist people with a disaster emergency response using a Flask Web App.


# Instructions

This project is divided in 3 sections:

1. Processing data, building an ETL pipeline to extract data from source, cleaning the data and saving it in a SQLite DB.
2. Build a machine learning pipeline that can classify tweet text messages into 36 different categories.
3. Running a web application which can show model results in real time.

# Dependencies

- Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

# Installation
To clone the git repository:

git clone https://github.com/abg3/Disaster-Response-ML-Project.git

# Running the application

You can run the following commands in the project's directory to set up the database, train the model and save the model.

To run ETL pipeline for cleaning data and store the processed data in the database:

Run the below command from the terminal-

- python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file:

Run the below command from the terminal-

- python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command in the app's directory to run your web app-

- python run.py

Open a browser and go to http://0.0.0.0:3001/. You can input any message and see the results.

# Notebooks

In the notebooks folder you can find two jupyter notebooks that will help you understand how the data was processes and model was trained.

ETL Preparation Notebook: About the implemented ETL pipeline
ML Pipeline Preparation Notebook: analyze the Machine Learning Pipeline developed with NLTK and Scikit-Learn

The ML Pipeline Preparation Notebook can be modified to re-train the model or tune it through a dedicated Grid Search section.

# Important Files

app/templates/*: templates/html files for web application

data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

models/train_classifier.py: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

app/run.py: This file is used to launch the Flask web app to classify disaster tweet messages.

disaster response.PNG: This is to see how the website looks like.

# Acknowledgements

Thank you Udacity and Figure Eight for providing the dataset and an opportunity to work on this project.

