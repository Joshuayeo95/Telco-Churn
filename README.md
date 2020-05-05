# Telco-Churn

This project aims to predict the probability of a customer churning from the Telco's platform. We will explore what are the different factors that influence a customers' decision to leave.

We may also do clustering to try and examine if there are different clusters of customers and explore if they have similar characteristics that make them more likely to churn. 

## Dataset

The dataset used for this project can be obtained from:
https://www.kaggle.com/blastchar/telco-customer-churn


## Modelling Analysis

As we are trying to classify if the customer if going to churn or not, we will be training and evaluating the following models:

* Random Forests
* Nearest Centroid
* Logistic Regression
* K-Nearest Neighbours

## Evaluation Metric

__F1 Score__ will be used as the evaluation metric in determining the models performance. The rationale is that we want a balance between the model's ability to identify as many potential churn customers (recall) and the costs of advertising to customers that had low propensity to churn (precision). 
