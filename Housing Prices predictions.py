import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotencoder


# READING DATA STAGE
X_train_full = pd.read_csv("C:\Users\nelly\OneDrive - University of Cape Town\Data Science\PERSONAL PROJECTS\Machine Learning\Housing Prices predictions\Data\train.csv", index_col = "Id")
X_test_full = pd.read_csv("C:\Users\nelly\OneDrive - University of Cape Town\Data Science\PERSONAL PROJECTS\Machine Learning\Housing Prices predictions\Data\test.csv", index_col = "Id")

#CLEANING DATA STAGE
X_train_full.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
y = X_train_full['SalePrice']
X_train_full.drop(['SalePrice'], axis = 1, inplace = True)

X_train_o, X_vaid_o, y_train, y_valid = train_test_split(X_train_full,y, train_size = 0.8, test_size = 0.2, random_state =0)

num_cols = [col for col in X_train_o.columns if X_train_o[col].dtype in ['float64','int64']]
cat_cols = [col for col in X_train_o.columns if X_train_o[col].dtype == ['object'] and X_train_o[col].nunique < 10]

my_cols = num_cols +cat_cols
X_train = X_train_o[my_cols].copy()
X_valid = X_valid_o[my_cols].copy()
X_test = X_test_full[my_cols].copy()


#PREPROCESSING STAGE

num_transformer = SimpleImputer(strategy = 'mean')
cat_transformer = Pipeline(steps = [('Impute', SimpleImputer(strategy = 'most_frequent'),('OH', OneHotEncoder(handle_unknown = 'ignore')])

transformer = ColumnTransformer(transformers = [('num', num_transformer, num_cols),('cat', cat_transformer, cat_cols)])

#MODEL DEFINITION, FITTING, AND PREDICTION
model = RandomForestRegressor(n_estimators = 100, random_state =0)

my_pipeline = Pipeline( steps = [('transformer', transformer),('model',model)])

my_pipeline.fit(X_train, y_train)

predictoions = my_pipeline.predict(X_valid)
print('MAE: ' + mean_absolute_error(y_vaid, predictions))








