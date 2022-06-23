# Disaster Response Pipeline Project
## Table of Contents

1. Summary of the project
2. How to run the Python scripts and web app
3. Explanation of the files

### Summary of the project:
---
In this Project, there is a data set containing real messages that were sent during disaster events and a machine learning pipeline will be created to categorize these events in order to send the messages to an appropriate disaster relief agency.

The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### How to run the Python scripts and web app:
---
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Explanation of the files
---
The project contains three main folders:
- data: From here, you can find the necessary datasets and also the python file for cleaning and saving data into a database
- models: This python file should help build a machine learning model which is applied to categorize the appropriate disasters.
- app: This python file should be run to generate the web app
