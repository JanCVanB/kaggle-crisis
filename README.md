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

![](http://artfcity.com/wp-content/uploads/2015/01/business-gif.gif)
