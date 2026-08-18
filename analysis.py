import pandas as pd


def load_data(file_path):

    # Load dataset
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove * from column names
    df.columns = (
        df.columns
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert Date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Numeric columns
    numeric_columns = [
        "Children apprehended and placed in CBP custody",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care"
    ]

    # Convert numeric columns
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

    # Remove rows where Date is missing
    df = df.dropna(subset=["Date"])

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def calculate_kpis(df):

    # Transfer Efficiency
    df["Transfer Efficiency (%)"] = (
        df["Children transferred out of CBP custody"]
        / df["Children apprehended and placed in CBP custody"]
    ) * 100

    # Discharge Effectiveness
    df["Discharge Effectiveness (%)"] = (
        df["Children discharged from HHS Care"]
        / df["Children in HHS Care"]
    ) * 100

    # Pipeline Throughput
    df["Pipeline Throughput"] = (
        df["Children transferred out of CBP custody"]
        + df["Children discharged from HHS Care"]
    )

    # CBP Backlog
    df["CBP Backlog"] = (
        df["Children in CBP custody"]
    )

    # HHS Backlog
    df["HHS Backlog"] = (
        df["Children in HHS Care"]
    )

    # Total Active Care Load
    df["Total Active Care Load"] = (
        df["Children in CBP custody"]
        + df["Children in HHS Care"]
    )

    return df


if __name__ == "__main__":

    file_path = "data/HHS_Unaccompanied_Alien_Children_Program.csv"

    # Load and clean data
    df = load_data(file_path)

    print("\n========== CLEAN DATASET ==========\n")

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nDate Range:")
    print(
        df["Date"].min().strftime("%d-%m-%Y"),
        "to",
        df["Date"].max().strftime("%d-%m-%Y")
    )

    # Calculate KPIs
    df = calculate_kpis(df)

    print("\n========== KPI RESULTS ==========\n")

    print(
        "Average Transfer Efficiency:",
        round(
            df["Transfer Efficiency (%)"].mean(),
            2
        ),
        "%"
    )

    print(
        "Average Discharge Effectiveness:",
        round(
            df["Discharge Effectiveness (%)"].mean(),
            2
        ),
        "%"
    )

    print(
        "Average Pipeline Throughput:",
        round(
            df["Pipeline Throughput"].mean(),
            2
        )
    )

    print(
        "Average CBP Backlog:",
        round(
            df["CBP Backlog"].mean(),
            2
        )
    )

    print(
        "Average HHS Backlog:",
        round(
            df["HHS Backlog"].mean(),
            2
        )
    )

    print(
        "Average Total Active Care Load:",
        round(
            df["Total Active Care Load"].mean(),
            2
        )
    )

    print("\n========== KPI PREVIEW ==========\n")

    print(
        df[
            [
                "Date",
                "Transfer Efficiency (%)",
                "Discharge Effectiveness (%)",
                "Pipeline Throughput",
                "CBP Backlog",
                "HHS Backlog",
                "Total Active Care Load"
            ]
        ].head(10)
    )