This project is for Kaggle competition. 
All information can e found using link https://www.kaggle.com/c/santander-customer-transaction-prediction

----------------
In this challenge, Kagglers should identify which customers will make a specific transaction in the future, 
irrespective of the amount of money transacted. 
The data provided for this competition has the same structure as the real data we have available to solve this problem.
----------------


Soution gives 170 place out of 8802 (top 2%).
The main idea is to use frequents as an additional features and then push them (+ original feature) into LGB model (8 folds).

The poject contains ClassLGB which can be used in another projects and quite convenient to transport. 

