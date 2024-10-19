import pandas as pd


def load_df(path):

    return pd.read_csv(path)


def clean_column_names(df):

    # aplicamos el map para limpiar los espacios
    cols = list(map(lambda col: col.replace(" ", ""), df.columns))

    # aplicamos el map para quitar los (
    cols = list(map(lambda col: col.split("(")[0] if "(" in col else col, cols))

    df.columns = cols

    return df


def extract_mobile_manufacturer(value):

    return value.split(" ")[0]


MANUFACTURER_MAPPING_DICT = {
    "Google": 1,
    "OnePlus": 2,
    "Samsung": 3,
    "Xiaomi": 4,
    "iPhone": 5,
}


def clean_df(path):

    df = load_df(path=path)

    df = (
        df.pipe(clean_column_names)
        .set_index("UserID")
        .assign(
            OperatingSystem=lambda df: (df["OperatingSystem"] == "iOS") * 1,
            Gender=lambda df: (df["Gender"] == "Male") * 1,
            DeviceModel=lambda df: df["DeviceModel"].apply(extract_mobile_manufacturer),
        )
        .assign(DeviceModel=lambda df: df["DeviceModel"].map(MANUFACTURER_MAPPING_DICT))
    )

    return df
