import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def prepare_features(df):

    df = df[df["preco_brl"] < 3000]

    df["log_price"] = np.log1p(df["preco_brl"])

    coords = df[["latitude", "longitude"]]

    kmeans = KMeans(n_clusters=20, random_state=42)

    df["location_cluster"] = kmeans.fit_predict(coords)

    return df


def build_features(df):

    numeric = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
    "latitude",
    "longitude",
    "location_cluster",
    "dist_copacabana",
    "dist_cristo",
    "dist_centro",
    "dist_beach"
]
    
    categorical = [
        "neighbourhood"
    ]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_cat = encoder.fit_transform(df[categorical])

    cat_cols = encoder.get_feature_names_out(categorical)

    X_cat = pd.DataFrame(X_cat, columns=cat_cols)

    X = pd.concat([df[numeric].reset_index(drop=True), X_cat], axis=1)

    y = df["log_price"]

    return X,y


def train_model(X,y):

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = XGBRegressor(random_state=42)

    param_grid = {

        "n_estimators":[300,500,700],
        "max_depth":[4,6,8],
        "learning_rate":[0.03,0.05,0.1]

    }

    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=10,
        cv=3
    )

    search.fit(X_train,y_train)

    model = search.best_estimator_

    preds = model.predict(X_test)

    real = np.expm1(y_test)

    pred = np.expm1(preds)

    mae = mean_absolute_error(real,pred)

    rmse = np.sqrt(mean_squared_error(real,pred))

    print("MAE:",mae)
    print("RMSE:",rmse)

    return model,X_test,y_test,preds


def shap_analysis(model,X):

    explainer = shap.Explainer(model)

    shap_values = explainer(X)

    shap.plots.bar(shap_values)


def error_map(df,y_test,preds):

    real = np.expm1(y_test)

    pred = np.expm1(preds)

    df = df.loc[y_test.index]

    df["error"] = abs(real-pred)

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="error",
        size="error",
        zoom=10
    )

    fig.update_layout(mapbox_style="carto-darkmatter")

    fig.show()


