
import pandas as pd

weather = pd.read_csv("sg-weather.csv", index_col="Date")
weather


weather.replace({"na": None, "N/A": None, "null": None}, inplace=True)
null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]



null_pct

weather.columns = weather.columns.str.lower()

weather


weather = weather.ffill()



weather.apply(pd.isnull).sum()


weather.dtypes


column_types = {
    "minimum relative humidity ": "int",
}

weather = weather.astype(column_types)


weather.dtypes


weather.index


weather.index = pd.to_datetime(weather.index)


weather.index.year.value_counts().sort_index()

weather["total rainfall "].plot()


weather = weather.sort_index()


weather["target"] = weather.shift(-1)["air temperature means daily maximum "]


weather

weather = weather.ffill()
weather



from sklearn.linear_model import Ridge #ridge regression (penalises coefficients for multicolinearity)


weather.corr()


rr = Ridge(alpha=.1)


predictors = weather.columns[~weather.columns.isin(["target"])]


predictors


#must use times series cross validation (cannot use future data to predict past data)

def backtest (weather, model, predictors, start=120, step= 6):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:] #take all data before current row 
        test = weather .iloc[i:(i+step),:] #take the next 6 months to make prediction on

        model.fit(train[predictors], train["target"]) 

        preds = model.predict(test[predictors]) #testing the model

        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1) #combining actual and predicted (axis=1 treats concated values as seperate columns)

        combined.columns = ["actual", "prediction"]

        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_predictions.append(combined)
    return pd.concat(all_predictions)


predictions = backtest(weather, rr, predictors)


predictions


from sklearn.metrics import mean_absolute_error

mean_absolute_error(predictions["actual"], predictions["prediction"])


def pct_diff(old, new):
    return (new - old) / old

def compute_rolling(weather, horizon, col): #find rolling averages, where horizons is number of months to compute the rolling average on
    label = f"rolling_{horizon}_{col}"

    weather[label] = weather[col].rolling(horizon).mean() #take last few rows before current row and comput the avg of a column across all those rows
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])

    return weather

rolling_horizons = [3, 6]

for horizon in rolling_horizons:
    for col in ["air temperature means daily maximum ", "air temperature means daily minimum ", "minimum relative humidity ", "highest daily rainfall total ", "24 hours mean relative humidity "]:
        weather = compute_rolling(weather, horizon, col)

weather

weather = weather.iloc[6:,:]

weather

weather = weather.fillna(0)

def expand_mean(df):
    return df.expanding(1).mean()

for col in ["air temperature means daily maximum ", "air temperature means daily minimum ", "minimum relative humidity ", "highest daily rainfall total ", "24 hours mean relative humidity "]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean) #group data by month and take all the temps before the date and calculate the avg temp

weather

predictors = weather.columns[~weather.columns.isin(["target"])]

predictors


predictions = backtest(weather, rr, predictors)


mean_absolute_error(predictions["actual"], predictions["prediction"])

predictions.sort_values("diff", ascending=False)


weather.loc["2004-06-01":"2005-6-01"]


predictions["diff"].round().value_counts().sort_index().plot()

import pickle

with open("predictions_model.pkl", "wb") as file:
    pickle.dump(predictions, file)
