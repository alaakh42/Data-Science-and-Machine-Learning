"""
Created on Fri Sep 30 15:29:10 2016

@author: Mis.Alla
"""
# 1- Clean Dataset
import pandas as pd

titanic = pd.read_csv('train.csv')
#print (titanic.describe())
# As we have some misssing values in Age column, we 'll fillna those NA slots with the medianof the Age column
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
#print (titanic.describe())

# All of the non-numeric columns don't give me so much information [ticket, Embarked,..]
# Except 'sex' column, but i will encode it with some numeric values instead: male '0' & female '1'
#print(titanic["Sex"].unique()) # print unique values of the 'Sex' column
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
#print (titanic.describe())

#lets do the same for Embarked column, but we will replace the missing values with the most common value which is 'S'
# then We'll assign the code 0 to S, 1 to C and 2 to Q
#print(titanic['Embarked'].unique())
titanic['Embarked'] = titanic['Embarked'].fillna('S')
#OR use this#titanic.loc[titanic['Embarked'] == 'nan', 'Embarked'] = 'S'
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

# 2- Train the Dataset using Linear Regression Algoithm
     # using cross-validation as a test

# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

#the column we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] # except "Survived" as it is the column we will predict
# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset. 
# It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kfold = KFold(titanic.shape[0], n_folds = 3, random_state = 1)

predictions = []
for train, test in kfold:
    # The predictors we're using the train the algorithm.
#Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic['Survived'].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)


# 3- Evaluating Error

import numpy as np
# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis = 0)
#Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions >  0.5] = 1
predictions[predictions <= 0.5] = 0
# accuracy = sum(predictions that equals the real value) divided by the total number of passengers
accuracy = sum(predictions[predictions == titanic["Survived"]])/ len(predictions)
print (accuracy*100) 

# 4- Logistic Regression

 # Linear regression is an input to the logistic regression, using 'the logit function' you cansqueeze the output between 0 and 1
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

#Initialize our algorithm
alg = LogisticRegression(random_state = 1)
#Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
score = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv = 3)
# Take the mean of the scores (because we have one for each fold)
print (score.mean()*100)

##Processing The Test Set
#
import pandas 
#
titanic_test = pandas.read_csv("titanic_test.csv")
titanic_test['Age'] = titanic_test['Age'].fillna(titanic['Age'].median())
titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1
titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')
titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0
titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1
titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())

# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("Titanic_submission1.csv")