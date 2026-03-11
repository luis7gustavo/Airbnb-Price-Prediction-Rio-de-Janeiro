import numpy as np
import plotly.express as px


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
        zoom=10,
        title="Mapa de Erro do Modelo"

    )

    fig.update_layout(mapbox_style="carto-darkmatter")

    fig.show()