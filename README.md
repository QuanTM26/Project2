# Disaster Response Pipeline Project
## Table of Contents

1. [Summary of the project](https://github.com/QuanTM26/Project2#summary-of-the-project)
2. [How to run the Python scripts and web app](https://github.com/QuanTM26/Project2#how-to-run-the-python-scripts-and-web-app)
3. [Explanation of the files](https://github.com/QuanTM26/Project2#explanation-of-the-files)

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
>data  
>|- disaster_categories.csv # data to process   
>|- disaster_messages.csv # data to process  
>|- process_data.py  
>|- DisasterResponse.db # database to save clean data to  

- models: This folder contains a saved model and a python file to build that machine learning model which is applied to categorize the appropriate disasters.
>models  
>|- train_classifier.py  
>|- classifier.pkl # saved model  

- app: This folder contains the python file and HTML file which should be run to generate the web app
>app  
>| - template  
>| |- master.html # main page of web app  
>| |- go.html # classification result page of web app  
>|- run.py # Flask file that runs app  

- README.md: this file shows some instructions to follow to run the scripts above
