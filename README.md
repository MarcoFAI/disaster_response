# Disaster Response Pipeline Project
 
### Project Motivation
In this project i analyzed data from disasters provided by [Figure Eight](https://appen.com/). The goal was to train a model that classifies whether a message is related to a disaster and, if so, classifies the type of disaster. With the trained model, emergency messages can be automatically forwarded to the appropriate disaster relief agencies. The project includes the preparation of training data, the training of a machine learning model and a web app in which the trained model can be used. Additionally, some visualizations can be found in the web app.

### File Descriptions
app    

| - template    
| |- master.html # main page of the web app    
| |- go.html # page showing the classification results  
|- run.py # Flask file that runs app    


data    

|- disaster_categories.csv # training data   
|- disaster_messages.csv # training data    
|- process_data.py # data preprocessing pipeline    
|- DisasterResponse.db # database where the cleaned data is saved to   


models   

|- train_classifier.py # machine learning pipeline     
|- model.pkl # saved model     


README.md    

### Components
The project consists of three components: 

#### 1. ETL Pipeline
In `process_data.py` an ETL pipeline is implemented. It loads the two datasets, puts them in context, cleans the data and stores it in a database.
 
#### 2. ML Pipeline
In the script 'train_classifier.py' a ML pipeline is implemented. It loads the cleaned data from the Database, splits it in training and test data, defines and builds a text processing and machine learning pipeline, trains a RandomForestClassifier using GridSearchCV, outputs the performance on the test data and exports the final model as a pickle file. 

#### 3. Flask Web App
The web app contains an input field in which the message to be classified can be entered. If you press the button next to it, the model classifies the message and a new page opens with the results of the classification. In addition, the web app shows some visualizations of the training data. 


### How to operate the project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/model.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to shown URL

### Licensing, Authors, Acknowledgements, etc.
Thanks to Udacity for helpful code for the web app and the data.
