import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.interpolate import UnivariateSpline
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from category_encoders import TargetEncoder

train_data_path = 'train.csv'
train_data = pd.read_csv(train_data_path)

y = np.log(train_data.SalePrice)

print(train_data.columns)

data_features = [ 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']

X = train_data[data_features]
encoder = TargetEncoder()
X= encoder.fit_transform(X, y)
#X = X.astype('category')

#lasso = LassoCV()
#lasso.fit(X, y)


#data_features = X.columns[lasso.coef_ != 0]
#X = X[data_features]

#print(f"Lasso 选择了 {len(data_features)} 个特征")

def fit_model(X, y,para_space=None):
    train_model = xgb.XGBRegressor(random_state=1,enable_categorical=True)
    if para_space is None:
        print("这是基础模型")
        train_model = train_model.fit(X,y)
        importance = train_model.feature_importances_
        feature_importance = pd.Series(importance, index=X.columns)
        #important_features = feature_importance[feature_importance > 0.001].index
        #print(feature_importance)
        feature_importance.to_csv("feature_importance.csv")


    else:
        print("这是贝叶斯优化")
        train_model = bayes(train_model,para_space)
        train_model = train_model.fit(X, y)
    return train_model


def get_mae(model):
    #val_predictions = model.predict(val_X)
    #my_mae = mean_absolute_error(val_y, val_predictions)
    my_mae = -np.mean(cross_val_score(
        model, X, y, scoring='neg_root_mean_squared_log_error', cv=5
    ))

    return my_mae

def bayes(base_rf,para_space):

    opt = BayesSearchCV(
    estimator=base_rf,
    search_spaces=para_space,
    n_iter=30,               # 迭代次数，也就是要搜索多少组参数
    cv=5,                    # 5折交叉验证
    scoring='neg_root_mean_squared_log_error',
    random_state=42,
    n_jobs=-1,                # 并行训练，-1 表示使用CPU所有核心
    n_points=1,
    )
    return opt

baseline_model = fit_model(X,y)
#base_n_leaves = max(est.get_n_leaves() for est in baseline_model.estimators_)
#base_depth = max(est.get_depth() for est in baseline_model.estimators_)
my_mae = get_mae(baseline_model)
print(f"base_rmse:{my_mae}")


para_space = {
    'n_estimators':Integer(100, 500),
    'max_depth':Integer(3, 10),
    'learning_rate':Real(0.01, 0.2, prior='log-uniform'),
    'subsample':Real(0.5, 1.0),
    'colsample_bytree':Real(0.5, 1.0),
    'gamma':Integer(0,5)
}

train_model = fit_model(X,y,para_space)
my_mae = get_mae(train_model)
print("Best Score (cv RMSE):", -train_model.best_score_)
print("Best Hyperparameters:", train_model.best_params_)


test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)
#test_y = test_data.SalePrice

test_X = test_data[data_features]
test_X= encoder.transform(test_X)

predict_log_y = train_model.best_estimator_.predict(test_X)
predict_y = np.exp(predict_log_y)
print(pd.Series(predict_y).head())

submission = pd.DataFrame({"Id":test_data["Id"],"SalePrice":predict_y})
submission.to_csv("submission.csv",index=False)
