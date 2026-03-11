import numpy as np


def haversine(lat1, lon1, lat2, lon2):

    R = 6371  # km

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def create_geo_features(df):

    # pontos turísticos principais

    copacabana = (-22.9711, -43.1822)
    cristo = (-22.9519, -43.2105)
    centro = (-22.9068, -43.1729)

    df["dist_copacabana"] = haversine(
        df["latitude"], df["longitude"], copacabana[0], copacabana[1]
    )

    df["dist_cristo"] = haversine(
        df["latitude"], df["longitude"], cristo[0], cristo[1]
    )

    df["dist_centro"] = haversine(
        df["latitude"], df["longitude"], centro[0], centro[1]
    )

    # ----------------------------------------------------
    # Pontos representando a costa/praias do Rio
    # ----------------------------------------------------

    beach_points = [

        (-22.9711, -43.1822),  # Copacabana
        (-22.9868, -43.1895),  # Ipanema
        (-22.9990, -43.2130),  # Leblon
        (-23.0035, -43.3200),  # Barra
        (-23.0300, -43.4650),  # Recreio
        (-23.0500, -43.5000)   # Prainha

    ]

    distances = []

    for lat_b, lon_b in beach_points:

        dist = haversine(
            df["latitude"],
            df["longitude"],
            lat_b,
            lon_b
        )

        distances.append(dist)

    distances = np.vstack(distances)

    df["dist_beach"] = distances.min(axis=0)

    return df