# Kaggle Crisis

Machine learning model to correlate financial report word frequencies with the 07-08 financial crisis

Created by Jan Van Bruggen, Yamei Ou, and Obinna Eruchalu
for the Kaggle competition at https://inclass.kaggle.com/c/cs-ee-155-kaggle-competition


## Kaggle Competition Instructions

In this competition, you are going to predict the appearance of the financial crisis era 
based on a collection of words extracted from thousands of quarterly and yearly reports of large financial service companies.

All the data sources are obtained from the U.S. Securities and Exchange Commission. 
Publicly traded companies are required by law to disclose information on an ongoing basis, 
and all large domestic issuers must submit annual reports on Form 10-K and quarterly reports on Form 10-Q. 
We downloaded those forms of most financial service companies in recent 10 years. 
On the page of Great Recession on Wikipedia, the U.S. Recession is defined in period between December 2007 and June 2009. 
We labeled '1' on the report if it was published during the U.S. Recession era and '0' otherwise.

The features you are going to have is a bag of words. 
These words are the top 500 most frequent word stems in the reports 
published in the U.S. Recession era with stop words filtered out. 
Your task is to reveal the correlation of the frequency of words and the appearance of the financial crisis.


## Our Model

We aggregated the out-of-sample predictions of multiple models to determine our final predictions.
First we used the word count and tf-idf datasets to train several different classifier models, including
Random Forest, AdaBoost, Decision Tree, Naive Bayes, and Support Vector Machine (both linear- and inifinite-kernel).
After examining the 5-fold cross-validation error of each dataset/classifier combination,
we decided that the best models to combine were
Random Forest (on word count), Random Forest (on tf-idf), and AdaBoost (on word count).
We aggregated these predictions with a simple majority vote,
and the aggregated predictions achieved 93% accuracy on the competition's final leaderboard.


![](http://i.imgur.com/ynqYgPw.gifv)
