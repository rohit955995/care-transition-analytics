import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def prepare_data(file_path):

    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove *
    df.columns = (
        df.columns
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    # Convert Date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    numeric_columns = [
        "Children apprehended and placed in CBP custody",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care"
    ]

    for column in numeric_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove invalid rows
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def train_model(df):

    features = [
        "Children apprehended and placed in CBP custody",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children discharged from HHS Care"
    ]

    target = "Children in HHS Care"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Feature importance
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        "Importance",
        ascending=False
    )

    return model, mae, rmse, r2, feature_importance


if __name__ == "__main__":

    file_path = "data/HHS_Unaccompanied_Alien_Children_Program.csv"

    df = prepare_data(file_path)

    model, mae, rmse, r2, feature_importance = train_model(df)

    print("\n========== MACHINE LEARNING MODEL ==========\n")

    print("Model: Random Forest Regressor")

    print("MAE:", round(mae, 2))

    print("RMSE:", round(rmse, 2))

    print("R² Score:", round(r2, 4))

    print("\n========== FEATURE IMPORTANCE ==========\n")

    print(feature_importance.to_string(index=False))

    print("\nModel training completed successfully!")