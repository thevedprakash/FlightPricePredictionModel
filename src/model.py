

from sklearn import linear_model
from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import SplineTransformer

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def regression(X_train, X_test, y_train, y_test):
    ## Baseline model
    reg = linear_model.LinearRegression()
    reg.fit(X_train,y_train)
    print(metrics.r2_score(y_train,reg.predict(X_train)), metrics.r2_score(y_test,reg.predict(X_test)))
    return reg

def knearestneighbour(X_train, X_test, y_train, y_test):
    k_range = list(range(1, 30))
    params = dict(n_neighbors = k_range)
    knn_regressor = GridSearchCV(KNeighborsRegressor(), params, cv =10, scoring = 'neg_mean_squared_error')
    knn_regressor.fit(X_train, y_train)
    print(metrics.r2_score(y_train,knn_regressor.predict(X_train)),metrics.r2_score(y_test,knn_regressor.predict(X_test)))
    return knn_regressor

def decisiontree(X_train, X_test, y_train, y_test):
    depth  =list(range(3,30))
    param_grid =dict(max_depth =depth)
    tree =GridSearchCV(DecisionTreeRegressor(),param_grid,cv =10)
    tree.fit(X_train,y_train)
    print(metrics.r2_score(y_train,tree.predict(X_train)),metrics.r2_score(y_test,tree.predict(X_test)))
    return tree

def randomForest(X_train, X_test, y_train, y_test):
    tuned_params = {'n_estimators': [100,  300,  500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    random_regressor = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter = 20, scoring = 'neg_mean_absolute_error', cv = 5, n_jobs = -1)
    random_regressor.fit(X_train, y_train)
    print(metrics.r2_score(y_train,random_regressor.predict(X_train)),metrics.r2_score(y_test,random_regressor.predict(X_test)))
    return random_regressor