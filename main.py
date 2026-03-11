from src.load_data import load_dataset
from src.clean_data import clean_price
from src.geo_features import create_geo_features
from src.train_model import run_training

from src.analysis import (
    converter_preco_para_reais,
    distribuicao_precos,
    distribuicao_tipos_imovel,
    preco_mediano_por_tipo,
    bairros_mais_caros,
    distribuicao_capacidade,
    mapa_densidade_airbnb
)


def main():

    print("\n=========== PROJETO AIRBNB DATA SCIENCE ===========\n")

    # --------------------------------------------------
    # 1. Carregar dataset
    # --------------------------------------------------

    print("Carregando dataset...\n")

    df = load_dataset("listings.csv.gz")

    print(f"Dataset carregado com {len(df)} registros\n")

    # --------------------------------------------------
    # 2. Selecionar colunas relevantes
    # --------------------------------------------------

    df = df[[
        "latitude",
        "longitude",
        "room_type",
        "property_type",
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "availability_365",
        "neighbourhood",
        "price"
    ]]

    # --------------------------------------------------
    # 3. Traduzir tipos de imóvel
    # --------------------------------------------------

    df["room_type"] = df["room_type"].replace({
        "Entire home/apt": "Casa/Apartamento inteiro",
        "Private room": "Quarto privado",
        "Shared room": "Quarto compartilhado",
        "Hotel room": "Quarto de hotel"
    })

    # --------------------------------------------------
    # 4. Limpar nomes de bairros
    # --------------------------------------------------

    bairros_rio = [
        "Copacabana", "Ipanema", "Leblon", "Botafogo", "Flamengo",
        "Leme", "Lagoa", "Gávea", "Barra Da Tijuca", "Recreio",
        "Centro", "Santa Teresa", "Catete", "Glória", "Urca",
        "São Conrado", "Vidigal", "Rocinha", "Jacarepaguá"
    ]

    def extrair_bairro(texto):

        texto = str(texto).title()

        for bairro in bairros_rio:
            if bairro in texto:
                return bairro

        return "Outros"

    df["neighbourhood"] = df["neighbourhood"].apply(extrair_bairro)

    # --------------------------------------------------
    # 5. Remover valores ausentes
    # --------------------------------------------------

    print("Removendo valores ausentes...\n")

    df = df.dropna()

    print(f"Registros após limpeza: {len(df)}\n")

    # --------------------------------------------------
    # 6. Limpar coluna de preço
    # --------------------------------------------------

    print("Limpando preços...\n")

    df = clean_price(df)

    # --------------------------------------------------
    # 7. Converter moeda USD → BRL
    # --------------------------------------------------

    print("Convertendo moeda...\n")

    df = converter_preco_para_reais(df)

    # --------------------------------------------------
    # 8. Remover outliers
    # --------------------------------------------------

    df = df[(df["preco_brl"] > 30) & (df["preco_brl"] < 3000)]

    # --------------------------------------------------
    # 9. Criar features geográficas
    # --------------------------------------------------

    print("Criando features geográficas...\n")

    df = create_geo_features(df)

    # --------------------------------------------------
    # 10. ANÁLISE EXPLORATÓRIA
    # --------------------------------------------------

    print("\nGerando gráficos de análise exploratória...\n")

    distribuicao_precos(df)

    distribuicao_tipos_imovel(df)

    preco_mediano_por_tipo(df)

    bairros_mais_caros(df)

    distribuicao_capacidade(df)

    mapa_densidade_airbnb(df)

    # --------------------------------------------------
    # 11. Treinar modelo
    # --------------------------------------------------

    print("\nTreinando modelo...\n")

    run_training(df)

    print("\n=========== PIPELINE FINALIZADO ===========\n")


if __name__ == "__main__":
    main()