# fraud_detect
A simple credit card fraud detection algorithm

The data has a massive class imbalance as all credit card fraud datasets do as fraud is rare. 
The data has a fraud class that represents 0.17% of all transactions.
To combat this we handle the class imbalance using SMOTE which essentially generates similar data to the fraud class for our gradient boosted trees
algorithm to perform better at detecting the data as now it synthetically has more data.
The link for the data set: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
