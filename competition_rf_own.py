import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import cross_val_score

train_data_path = 'train.csv'
train_data = pd.read_csv(train_data_path)

y = train_data.SalePrice



#print(train_data.columns)

data_features = [
    'MSSubClass',
    'LotArea',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'YearRemodAdd',
    '1stFlrSF',
    '2ndFlrSF',
    'LowQualFinSF',
    'GrLivArea',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'Fireplaces',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    '3SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'MiscVal',
    'MoSold',
    'YrSold']

X = train_data[data_features]


test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)
#test_y = test_data.SalePrice

test_X = test_data[data_features]


#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


def fit_model(X, y, max_leaf_nodes=None):
    train_model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    train_model.fit(X, y)
    return train_model


def get_mae(model):
    #val_predictions = model.predict(val_X)
    #my_mae = mean_absolute_error(val_y, val_predictions)
    my_mae = -np.mean(cross_val_score(
        model, X, y, scoring='neg_root_mean_squared_log_error', cv=5
    ))
    return my_mae


def comput_xy(min_leaf_nodes, max_leaf_nodes,step): #生成difine对应mae
    tree_leaf = np.arange(min_leaf_nodes, max_leaf_nodes, step)
    mae_values = []
    for max_leaf_nodes in tree_leaf:
        train_model = fit_model(X,y, max_leaf_nodes)
        my_mae = get_mae(train_model)
        mae_values.append(my_mae)
    #print(f"mae_value:{mae_values}")

    return np.array(tree_leaf), np.array(mae_values) #转np data形式


def gradients(tree_leaf, mae_values): #
    print(f"二阶段tree_leaf:{tree_leaf}")
    gradients = np.gradient(mae_values, tree_leaf)#求该函数导数，输出np data（array）
    print(f"gradients:{gradients}")
    zero_index = np.where(np.diff(np.sign(gradients)))[0] #输出array，then[0]转为list)
    print(f"zero_index:{zero_index}")
    min_mae = mae_values[zero_index[0]]
    best_leaf_nodes = tree_leaf[zero_index[0]]
    for i in zero_index:
        i = int(i)
        if mae_values[i]<min_mae:
            min_mae = mae_values[i]
            best_leaf_nodes = tree_leaf[i]

    return best_leaf_nodes


def Uspline(tree_leaf,mae_values): #一阶段粗调返回redefine
    spline = UnivariateSpline(tree_leaf, mae_values,k=4, s=0) #创建可调用对象
    spline_derivation = spline.derivative()
    zero_point = spline_derivation.roots() #返回np数组array，导函数的根=极值点x值
    print(f"zero_point:{zero_point}")
    if len(zero_point) == 0:
        best_leaf_nodes = tree_leaf[np.argmin(mae_values)]
    else:
        for i in zero_point:
            best_leaf_nodes = min(zero_point, key = lambda x:spline(x))#np.float64

    print(f"best_leaf_nodes:{best_leaf_nodes}")
    print(f"tree_leaf:{tree_leaf}")
    best_index = np.where(np.isclose(zero_point, best_leaf_nodes))[0][0]
    print(f"best_index:{best_index}")
    start_idx = max(0, best_index - 1)
    end_idx = min(len(tree_leaf) - 1, best_index + 1)
    tree_leaf_list = list(range(round(zero_point[start_idx]), round(zero_point[end_idx] + 1)))
    print(f"tree_leaf_list:{tree_leaf_list}")

    return tree_leaf_list


def final_model(best_leaf_nodes):
    final_model = fit_model(X, y, round(best_leaf_nodes))
    min_mae = get_mae(final_model)
    print(f'best_leaf_nodes:{best_leaf_nodes}, min_mae:{min_mae}')

    final_model = fit_model(X, y, round(best_leaf_nodes))
    predict_y = final_model.predict(test_X)
    print(pd.Series(predict_y).head())
    return final_model


def prediction():
    pass



baseline_model = fit_model(X,y)
base_n_leaves = max(est.get_n_leaves() for est in baseline_model.estimators_)
my_mae = get_mae(baseline_model)
print(f"base_mae:{my_mae}")

tree_leaf, mae_values = comput_xy(2, base_n_leaves,5)
tree_leaf_list = Uspline(tree_leaf,mae_values)
tree_leaf, mae_values = comput_xy(tree_leaf_list[0],tree_leaf_list[-1],1)
best_leaf_nodes = gradients(tree_leaf, mae_values)
final_model(best_leaf_nodes)


#predict_y = final_model.best_estimator_.predict(test_X)
#print(pd.Series(predict_y).head())
