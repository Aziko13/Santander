# Santander

This project is for Kaggle competition. The problem is to identify which customers 
will make a specific transaction in the future, irrespective of the amount of money transacted. 
All details can be found [here] [https://www.kaggle.com/c/santander-customer-transaction-prediction]

## Getting Started

Solution gives 170 place out of 8802 (top 2%, silver). The main idea is to use values frequency as 
additional features and then push them (+ original feature) into LGB model (8 folds).
The poject contains ClassLGB which can be used in another projects and quite transportable. 

### Prerequisites

* Python 3.7.3
* lightgbm 2.1.0
* numpy 
* pandas


### Project's structure:
	├── input
	├── kfold
	├── model_predictions
	├── model_source
	├── src
		├── classLGB.py
		├── final_model.py
		├── params_tuning.py
		├── test.txt
	├── submissions


## Authors

* **Aziz Abdraimov** - *Initial work* - [Aziko13](https://github.com/Aziko13)

