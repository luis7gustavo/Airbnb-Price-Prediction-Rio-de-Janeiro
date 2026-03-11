from src.model import (
    prepare_features,
    build_features,
    train_model
)

from src.model_visualization import (
    grafico_real_vs_previsto,
    distribuicao_erro,
    importancia_variaveis
)

import numpy as np


def run_training(df):

    df = prepare_features(df)

    # manter apenas os dois tipos principais
    tipos_modelo = [
        "Casa/Apartamento inteiro",
        "Quarto privado"
    ]

    df = df[df["room_type"].isin(tipos_modelo)]

    for tipo in tipos_modelo:

        print("\n==============================")
        print("Treinando modelo para:", tipo)
        print("==============================")

        subset = df[df["room_type"] == tipo]

        if len(subset) < 50:
            print("Poucos dados, pulando...")
            continue

        X, y = build_features(subset)

        modelo, X_test, y_test, preds = train_model(X, y)

        preco_real = np.expm1(y_test)
        preco_previsto = np.expm1(preds)

        # gráficos do modelo
        grafico_real_vs_previsto(preco_real, preco_previsto)

        distribuicao_erro(preco_real, preco_previsto)

        importancia_variaveis(modelo, X.columns)