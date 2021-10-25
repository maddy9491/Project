import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

X = pd.read_csv('E:\\machine-learning-\\py-master\\home-data-for-ml-course\\train.csv')
X_test = pd.read_csv("E:\\machine-learning-\\py-master\\home-data-for-ml-course\\test.csv")

X = X.drop(['FireplaceQu', 'LandContour', 'GarageCond', 'Electrical', 'Fence', 'GarageQual', 'Street', 'Alley', 'Utilities', 'LandSlope', 'BsmtFinType2', 'BsmtFinSF2', 'LowQualFinSF', 'PoolQC', 'MiscFeature', 'MiscVal'], axis='columns')
X_test = X_test.drop(['FireplaceQu', 'LandContour', 'GarageCond', 'GarageQual', 'Electrical', 'Fence', 'Street', 'Alley', 'Utilities', 'LandSlope', 'BsmtFinType2', 'BsmtFinSF2', 'LowQualFinSF', 'PoolQC', 'MiscFeature', 'MiscVal'], axis='columns')
X = X.drop(['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'RoofMatl', 'ExterCond', 'Heating'], axis='columns')
X_test = X_test.drop(['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'RoofMatl', 'ExterCond', 'Heating'], axis='columns')


X.LotFrontage = X.LotFrontage.fillna(X.LotFrontage.mean())
X_test.LotFrontage = X_test.LotFrontage.fillna(X_test.LotFrontage.mean())
X.MasVnrArea = X.MasVnrArea.fillna(X.MasVnrArea.mean())
X_test.MasVnrArea = X_test.MasVnrArea.fillna(X_test.MasVnrArea.mean())
X.MasVnrArea = X.MasVnrArea.fillna(X.MasVnrArea.mean())
X_test.MasVnrArea = X_test.MasVnrArea.fillna(X_test.MasVnrArea.mean())
X.GarageYrBlt = X.GarageYrBlt.fillna(method='ffill')
X_test.GarageYrBlt = X_test.GarageYrBlt.fillna(method='ffill')
X.GarageType = X.GarageType.fillna(method='ffill')
X_test.GarageType = X_test.GarageType.fillna(method='ffill')
X.GarageFinish = X.GarageFinish.fillna(method='ffill')
X_test.GarageFinish = X_test.GarageFinish.fillna(method='ffill')


y = X.SalePrice
X = X.drop(['SalePrice'], axis='columns')

le = LabelEncoder()

X_train_obless = X.select_dtypes(exclude=['object'])
X_train_obwith = X.select_dtypes(include=['object'])
X_test_obless = X.select_dtypes(exclude=['object'])
X_test_obwith = X.select_dtypes(include=['object'])

X = X.fillna(0)


X = X.drop(X_train_obwith, axis='columns')
X_test = X_test.drop(X_test_obwith, axis='columns')

for cols in X_train_obwith.columns:
    X_train_obwith[cols] = le.fit_transform(X_train_obwith[cols])

for cols in X_test_obwith.columns:
    X_test_obwith[cols] = le.fit_transform(X_test_obwith[cols])

final_X = pd.concat([X, X_train_obwith], axis=1)
final_X_test = pd.concat([X_test, X_test_obwith], axis=1)

final_X_test = final_X_test.fillna(0)
model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(final_X, y)

predictions = model.predict(final_X_test)

output = pd.DataFrame({'Id': final_X_test.index,
                       'SalePrice': predictions})

final_X.to_csv('blah.csv', index=False)
output.to_csv('Submission.csv', index=False)
final_X_test.to_csv('test.csv', index=False)
