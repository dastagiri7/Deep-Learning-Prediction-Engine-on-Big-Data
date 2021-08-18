from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sb

from pyjavaproperties import Properties
properties = Properties()
properties.load(open('/home/giri/global.properties'))

from hdfs import InsecureClient
client_hdfs = InsecureClient(properties.getProperty('HDFS_CLIENT'))

# Data Loading from HDFS to local Dataframes

with client_hdfs.read(properties.getProperty('TRAIN_HDFS'), encoding = 'utf-8') as reader:   #try to retrieve by chunk_size
    train = pd.read_csv(reader)

with client_hdfs.read(properties.getProperty('TEST_HDFS'), encoding = 'utf-8') as reader:
    test = pd.read_csv(reader)

###Normalization

# combine all data to do preprocessing
combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop(['index', 'Id'], inplace=True, axis=1)


# Differentiate the columns with and without Na's
def columnDifferncesWithNa(df,dtype):
    if (dtype == 'number'):
        temp = df.select_dtypes(exclude=['object'])
    elif (dtype == 'string'):
        temp = df.select_dtypes(include=['object'])
    colsWithNoNa = []
    colsWithNa = []
    for col in temp.columns:
        if not df[col].isnull().any():
            colsWithNoNa.append(col)
        if df[col].isnull().any():
            colsWithNa.append(col)
    return colsWithNoNa, colsWithNa

intCols_NoNa, intCols__Na = columnDifferncesWithNa(combined , 'number')
strCols_NoNa, strCols_Na = columnDifferncesWithNa(combined , 'string')
print('Int cols without Na: ',len(intCols_NoNa), len(intCols__Na))
print('Str cols without Na',len(strCols_NoNa), len(strCols_Na))



# BsmtFullBath + FullBath = TotFullBath  : custom
# BsmtHalfBath + HalfBath = TotHalfBath  : custom
combined['BsmtFullBath'] = combined['BsmtFullBath'].fillna(0)
combined['BsmtHalfBath'] = combined['BsmtHalfBath'].fillna(0)

combined['TotFullBath'] = combined['BsmtFullBath'] + combined['FullBath']
combined['TotHalfBath'] = combined['BsmtHalfBath'] + combined['HalfBath']

# OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch = TotPorchArea : custom
combined['TotPorchArea'] = combined['OpenPorchSF'] + combined['EnclosedPorch'] + combined['3SsnPorch'] + combined['ScreenPorch']

# OverallCond + OverallQual = OverAllGrade : custom
combined['OverAllGrade'] = combined['OverallCond'] + combined['OverallCond']

#GarageArea, GarageCars
combined['GarageArea'] = combined['GarageArea'].fillna(0)
combined['GarageCars'] = combined['GarageCars'].fillna(0)
#arageCond
combined['GarageCond'] = combined['GarageCond'].fillna('NoGarage')

#TotalBsmtSF
combined['TotalBsmtSF'] = combined['TotalBsmtSF'].fillna(0)
#BsmtCond
combined['BsmtCond'] = combined['BsmtCond'].fillna('NoBsmt')

#MSZoning : with RL because this is the most frequent
combined['MSZoning'] = combined['MSZoning'].fillna('RL')

#SaleType : only one Na and replace with the most frequent one
combined['SaleType'] = combined['SaleType'].fillna('WD')


# LotFrontage: more NaN's
# Street : More pave : No effect on model
# Alley : more NaN's
# Utilities : all AllPub : No effect
# MasVnrType : more None's
# MasVnrArea : most are 0.0
# BsmtFinType1, BsmtFinSF1, BsmtFinType2, BsmtFinSF2 :
# Heating : almost same type except for 32 in 1460
# 1stFlrSF + 2ndFlrSF + LowQualFinSF = GrLivArea   (observation1)
# BsmtFullBath + FullBath = TotFullBath  : custom
# BsmtHalfBath + HalfBath = TotHalfBath  : custom
# OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch = TotPorchArea : custom
# PoolArea and PoolQC : Because only 0.5% data is filled : time waste
# Fence : more Na's
# MiscFeature : more Na's
# Functional : most of the data is similar : no effect
# OverallCond + OverallQual = OverAllGrade : custom

combined.drop(['LotFrontage', 'Street', 'Alley', 'Fence', 'Utilities', 'MasVnrType', 'MasVnrArea', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2','Heating', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'Functional', 'Electrical', 'Exterior1st', 'Exterior2nd', 'FireplaceQu', 'BsmtUnfSF', 'GarageYrBlt', 'BsmtExposure', 'BsmtQual', 'GarageFinish', 'GarageQual', 'GarageType', 'KitchenQual', 'OverallCond', 'OverallQual'], axis=1, inplace=True)

intCols_NoNa, intCols__Na = columnDifferncesWithNa(combined , 'number')
strCols_NoNa, strCols_Na = columnDifferncesWithNa(combined, 'string')
print(intCols_NoNa)
print(strCols_NoNa)
print('Int cols without Na: ',len(intCols_NoNa), len(intCols__Na))
print('Str cols without Na',len(strCols_NoNa), len(strCols_Na))



# to change string categorical data to seq of associated numbers (one hot encoded)
# and remove actual column
def categoricalDataToDummies(temp, cols):
    for col in cols:
        if(temp[col].dtype == np.dtype('object')):
            cat = pd.get_dummies(temp[col], prefix=col)
            temp = pd.concat([temp, cat], axis=1)
            temp.drop([col], axis = 1, inplace=True) #made other cat cols so remove
    return temp

combined = categoricalDataToDummies(combined, strCols_NoNa)


# data division
trainX = combined[:1460]
testX = combined[1460:]


# to remove outliers for continuous data LotArea, GrLivArea, TotPorchArea (Observations)

column = 'LotArea'
tr = pd.concat([trainX['SalePrice'], trainX[column]], axis=1)
tr.plot.scatter(x=column, y='SalePrice', ylim=(0,800000));

# outliers removal
trainX = trainX.drop(trainX[(trainX['LotArea']>100000)].index)
trainX = trainX.drop(trainX[(trainX['GrLivArea']>4000)].index)
trainX = trainX.drop(trainX[(trainX['TotPorchArea']>800)].index)  #>600

# Skewness abservations
# GrLivArea, SalePrice, TotPorchArea

sb.distplot(trainX['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(trainX['SalePrice'], plot=plt)  #df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF']


trainX.loc[trainX['SalePrice']>0, 'SalePrice'] = np.log1p(trainX['SalePrice'])  #

combined.head(10).to_csv('/home/giri/DemoData/combined10.csv', index=False) # just for safety


# Modelling

# target variable
trainY = trainX.SalePrice
# drop from train and test
testX.drop(['SalePrice'], inplace=True, axis=1)
trainX.drop(['SalePrice'], inplace=True, axis=1)


# Deep lerning neural network by usimg Keras (tensorflow as bottom)
deepModel = Sequential()

deepModel.add(Dense(128, kernel_initializer='normal', input_dim = trainX.shape[1], activation='relu')) # intput layer: 1
deepModel.add(Dense(256, kernel_initializer='normal',activation='relu')) # hidden layers: 3
deepModel.add(Dense(256, kernel_initializer='normal',activation='relu'))
deepModel.add(Dense(256, kernel_initializer='normal',activation='relu'))
deepModel.add(Dense(1, kernel_initializer='normal',activation='linear')) # output layer: 1

deepModel.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error']) # compilation state
#deepModel.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error']) # compilation state
deepModel.summary() #

# to makewhen to stop the callbacks nested
checkpoint = ModelCheckpoint('DeepLearingKeras', monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

# fit the deepModel to traing dependent and independent var's
deepModel.fit(trainX, trainY, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


# preditions
predictions = deepModel.predict(testX)

# reverse log
predictions = np.exp(predictions)

# file gen
dnn = pd.DataFrame({'Id':test.Id,'SalePrice':predictions[:,0]})
#dnn.to_csv('/home/giri/DemoData/keggle/DNN.csv',index=False)


# Writing Dataframe to hdfs
with client_hdfs.write('/DL/dnn_res.csv', encoding = 'utf-8') as writer:
 dnn.to_csv(writer)


"""
reading parameters from local configuration file /home/giri/global.properties

HDFS_CLIENT=http://localhost:9870
TRAIN_HDFS=/DL/train.csv
TEST_HDFS=/DL/test.csv
RESULT_HDFS=

"""
