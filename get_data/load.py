# -*- coding: utf-8 -*-
# @Time : 2018/5/2 9:21
# @Author : chen
# @Site : 
# @File : load.py
# @Software: PyCharm

'''
数据集的结构
longitude，latitude：经纬度
housing_median_age: 房屋年龄的中位数
total_rooms: 总房间数
total_bedrooms: 卧室数量
population: 人口数
households: 家庭数
median_income: 收入中位数
median_house_value: 房屋价值中位数
ocean_proximity: 离大海的距离
'''

import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

# 1. 取数据集
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    '''
    获取数据集
    '''
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    '''
    利用pandas读取CSV文件，返回一个相应的数据类型
    '''
    csv_path = os.path.join('./', housing_path, "housing.csv")
    data = pd.read_csv(csv_path)
    # print(data)
    return data

data = load_housing_data()

# df.value_counts() 可以帮助我们统计每一列数据的分布情况
# df.describe() 可以帮助我们整体了解数据集的情况，包含count,mean,min,max,std(标准差)等等
print(data["ocean_proximity"].value_counts())
print(data.describe())

import matplotlib.pyplot as plt

# hist()画出每个数值属性的柱状图
# data.hist(bins=50, figsize=(20, 15))
# plt.show()

# 2. 划分测试集与训练集（随机取样）
import numpy as np
# 2.1 自己写的划分方法
def split_train_test(data, test_ratio):
    '''
    对数据划分训练集与测试集
    :param data: 原始数据
    :param test_ratio: 测试集的大小
    :return: (训练集, 测试集)
    '''
    shuffled_indices = np.random.permutation(len(data)) # permutation(x)将x中的打乱重排
    test_set_size = round(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

'''
这里考虑到了一个问题：
如果再次运行程序，就会产生不同的测试集，多次运行后，你就会得到整个数据集
解决方法：1）设置随机种子；2）保存训练集与测试集
但是这两种方法都是一种静态的方法，党员是数据发生变化的时候，这些方法将出现问题
所以最后采用对每个实例赋予识别码，判断实例是否应该放入测试集
新的测试集将包含新实例中的20%，但是不会有之前在训练集的实例
'''
import hashlib
'''
下面的这个方法是利用了哈希索引的思想
将行号放入哈希表中进行判断（重点是256*0.2这个理解）,256的来源为2^8==256，因为这是取最后一个字节
'''
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    '''
    本函数是用来通过某一列进行训练集与测试集的划分工作
    :param data: 原始数据集
    :param test_ratio: 测试集的大小
    :param id_column: 划分依照的列的属性
    '''
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash)) # 这个id_我还是没明白是怎么回事

    return data.loc[~in_test_set], data.loc[in_test_set] # 细节使用~

housing_with_id = data.reset_index() # reset_index()增加一列作为索引"index"，作为行索引
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# 但考虑到数据集发生变化的话，行索引就会出现问题，我们就采用经纬度进行标记
housing_with_id["id"] = data["longitude"] * 1000 + data["latitude"]
temp_column = housing_with_id.pop("id")
housing_with_id.insert(0, "id", temp_column) # 这个就是用来在第0列插入一个索引
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# 2.2 sklearn自带的方法
from  sklearn.model_selection import train_test_split # sklearn中内置的数据集取样的方法

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)


# 3. 分层抽样
'''
前提是专家告诉我们，收入中位数是预测房价中位数非常重要的属性，故我们进行数据集的划分时要考虑到收入中位数
1）首先创建收入类别属性
2）然后针对收入分层的结果进行划分（每个收入类选择20%作为测试集）
'''
data["income_cat"] = np.ceil(data["median_income"] / 1.5) # ceil是进行数据取舍，以产生离散的分类
# 注：inplace的含义就是如果True则是在原DataFrame上面进行相关的操作；如果是False就是新形成一个DataFrame保留原本的DataFrame
data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True) # where实际上可以看成一个for循环与if else块的合成
# print(data.ix[:, ["id", "income_cat"]])

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["income_cat"]): # 这段代码就是分层获取训练集与测试集的索引
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

print(data["income_cat"].value_counts() / len(data)) # 查看分层的大小
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


housing = strat_train_set.copy()
# 画出坐标的散点图
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1) # 这个就是画散点图
# plt.show() # 在pycharm一定要加这个不然图像显示不了
# 画出每个点的房价情况，s：表示人口数量；c：表示该地区房价的中位数；cmap：用来定义颜色
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
plt.legend()
# plt.show()

# 4. 测算各个属性之间的关系系数
corr_matrix = housing.corr() # 初始化一个实例，corr()使用的是Pearson相关系数
print(corr_matrix["median_house_value"].sort_values(ascending=False)) # 调用median_house_value这个属性测算其他属性于这个属性之间的系数关系

from pandas.plotting import scatter_matrix # 这个案例上写错了，scatter_matrix表示画出attributes中个属性之间的关系

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
# plt.show()

# 5. 属性组合实验
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


# 6. 数据处理
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# 6.1 数据清洗
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)


from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median") # 这个strategy有三种median/mean/most_frequent

housing_num = housing.drop("ocean_proximity", axis=1) # 删除ocean_proximity这个参数因为它本身不是一个数值
imputer.fit(housing_num) # 重点的方法，fit()方法将实例拟合到训练数据中

all_median = imputer.statistics_ # 所有数据的中间值


X = imputer.transform(housing_num) # 重点的方法，transform()将中值补充进入缺失值，这是个数组
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# 6.2 处理文本
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder() # 这个是用来将字符串编码为数字（数字的选取都在字符串的长度-1）
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat) # 这里有两个fit，fit+transform就等于fit_transform
print(housing_cat_encoded)

# 6.3 处理文本（使用one_hot）
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
# 注意这里要求使用reshape(1, -1),原因在于fit_transform()需要传入一个2D的数组
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(1, -1)) # 这个结果是一个稀疏矩阵
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat) # 一步转换
print(housing_cat_1hot)

# 6.4 自定义文本转化器
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    '''
    仔细看了下这个类，其实就是用来增加三个或者两个自定义的属性，便于分析
    主要是将步骤5中的组合属性用到了这个地方
    '''
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            # 用np.c_()将XX，YY拉平后的两个array按照列合并(此时是n*2的举证，有n个样本点，每个样本点有横纵2维),然后调用分类器集合的decision_function函数获得样本到超平面的距离。Z是一个n*1的矩阵(列向量)，记录了n个样本距离超平面的距离。
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# 6.5 缩放处理（标准化）
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
'''
这里使用了相当重要的pipeline这个方法
pipeline就是一个流水线的作业方式
'''
num_pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    sklearn没有工具来处理Pandas DataFrame，所以要写一个
    简单的自定义转换器来做这个工作
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class MyLabelBinarizer(TransformerMixin): # 如果不写这个转换公式就会报错，这个切记
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

from sklearn.pipeline import FeatureUnion
'''
*****利用FeatureUnion将多个pipeline合在一起，
*****充分利用流水线进行相关的操作
'''

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', MyLabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(X=housing)
housing_temp = pd.DataFrame(housing_prepared) # 因为这个columns已经更新了不能使用之前的columns
print(housing_prepared)


# 7. 线性规划
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression() # 创建一个回归的实例
lin_reg.fit(housing_prepared, housing_labels) # 拟合数据集分别是训练集以及训练标签

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data) # 调用full_pipeline对数据进行预处理
print("Predictions:\t", lin_reg.predict(some_data_prepared)) # 调用predict()进行预测

print("Labels:\t\t", list(some_labels))

# 7.1 计算误差值（rmse）
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse) # 68628.19819848923，欠拟合

# 8 决策树回归
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_predictions, housing_labels)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse) # 0.0 ，过拟合

# 9 交叉验证进行评估
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    '''

    Mean: 71308.84564986329
    Standard deviation: 2768.2327970349124

    Mean: 53035.090742502085
    Standard deviation: 2233.615856387498
    '''
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(rmse_scores)

# 10 随机森林回归
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_predictions, housing_labels)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 5]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# 11 用测试机评估系统
final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("median_house_value", axis=1) # 去掉测试集的标签值
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)

final_predicitons = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(final_predicitons, y_test)
final_rmse = np.sqrt(final_mse) # final_rmse = 47736.35981348409
print(final_rmse)