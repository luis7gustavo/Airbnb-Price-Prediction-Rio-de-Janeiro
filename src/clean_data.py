def clean_price(df):

    df["price"] = (
        df["price"]
        .replace("[\$,]", "", regex=True)
        .astype(float)
    )

    return df


def remove_missing(df):

    df = df.dropna(subset=["price"])

    return df


def remove_outliers(df):

    q1 = df["price"].quantile(0.25)
    q3 = df["price"].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df = df[(df["price"] >= lower) & (df["price"] <= upper)]

    return df