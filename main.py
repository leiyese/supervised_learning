# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from load_data import csv_to_df

df_test = csv_to_df("home-data-for-ml-course/test.csv")
df_train = csv_to_df("home-data-for-ml-course/train.csv")

df_test.head(), df_train.head()
# %%
{
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {
                "language": "markdown"
            },
            "source": [
                "# Data Preprocessing för House Price Prediction",
                "",
                "Vi ska genomföra följande steg:",
                "1. Ladda och inspektera data",
                "2. Hantera saknade värden",
                "3. Feature engineering och transformationer",
                "4. Skalning av data"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {
                "language": "python"
            },
            "source": [
                "# Importera nödvändiga bibliotek",
                "import numpy as np",
                "import pandas as pd",
                "import matplotlib.pyplot as plt",
                "import seaborn as sns",
                "from sklearn.preprocessing import StandardScaler",
                "",
                "# Ladda data",
                "df_train = pd.read_csv('home-data-for-ml-course/train.csv')",
                "df_test = pd.read_csv('home-data-for-ml-course/test.csv')",
                "",
                "print('Träningsdata shape:', df_train.shape)",
                "print('Testdata shape:', df_test.shape)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {
                "language": "python"
            },
            "source": [
                "# Visa information om datatyperna och saknade värden",
                "print(\"TRÄNINGSDATA INFORMATION:\")",
                "print(\"\\nKolumner och datatyper:\")",
                "print(df_train.dtypes)",
                "",
                "print(\"\\nSaknade värden i procent:\")",
                "missing_train = (df_train.isnull().sum() / len(df_train)) * 100",
                "print(missing_train)",
                "",
                "print(\"\\n\\nTESTDATA INFORMATION:\")",
                "print(\"\\nKolumner och datatyper:\")",
                "print(df_test.dtypes)",
                "",
                "print(\"\\nSaknade värden i procent:\")",
                "missing_test = (df_test.isnull().sum() / len(df_test)) * 100",
                "print(missing_test)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {
                "language": "python"
            },
            "source": [
                "# Visualisering av saknade värden",
                "plt.figure(figsize=(15, 12))",
                "",
                "plt.subplot(2, 1, 1)",
                "plt.title('Saknade värden i träningsdata')",
                "missing_train.plot(kind='bar')",
                "plt.xticks(rotation=90)",
                "",
                "plt.subplot(2, 1, 2)",
                "plt.title('Saknade värden i testdata')",
                "missing_test.plot(kind='bar')",
                "plt.xticks(rotation=90)",
                "",
                "plt.tight_layout()",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {
                "language": "python"
            },
            "source": [
                "# Funktion för att rengöra dataset enligt reglerna",
                "def clean_dataset(df):",
                "    # Ta bort Id-kolumnen",
                "    if 'Id' in df.columns:",
                "        df = df.drop('Id', axis=1)",
                "    ",
                "    # Ta bort kolumner med mer än 40% saknade värden",
                "    missing_percentages = df.isnull().sum() / len(df) * 100",
                "    columns_to_drop = missing_percentages[missing_percentages > 40].index",
                "    df = df.drop(columns_to_drop, axis=1)",
                "    ",
                "    # Separera numeriska och kategoriska kolumner",
                "    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns",
                "    categorical_columns = df.select_dtypes(include=['object']).columns",
                "    ",
                "    # Behåll endast kategoriska kolumner med mindre än 5% saknade värden",
                "    categorical_missing = df[categorical_columns].isnull().sum() / len(df) * 100",
                "    categorical_to_keep = categorical_missing[categorical_missing < 5].index",
                "    ",
                "    # Kombinera kolumner att behålla",
                "    columns_to_keep = list(numeric_columns) + list(categorical_to_keep)",
                "    ",
                "    # Viktiga kolumner som vi alltid vill behålla",
                "    important_columns = ['GrLivArea', 'OverallQual', 'GarageArea', 'TotalBsmtSF', ",
                "                        'YearBuilt', 'LotArea']",
                "    ",
                "    for col in important_columns:",
                "        if col in df.columns and col not in columns_to_keep:",
                "            columns_to_keep.append(col)",
                "    ",
                "    return df[columns_to_keep]",
                "",
                "# Applicera rengöring på både tränings- och testdata",
                "df_train_cleaned = clean_dataset(df_train.copy())",
                "df_test_cleaned = clean_dataset(df_test.copy())",
                "",
                "print(\"Träningsdata shape före rengöring:\", df_train.shape)",
                "print(\"Träningsdata shape efter rengöring:\", df_train_cleaned.shape)",
                "print(\"\\nTestdata shape före rengöring:\", df_test.shape)",
                "print(\"Testdata shape efter rengöring:\", df_test_cleaned.shape)",
                "",
                "print(\"\\nKvarvarande kolumner:\")",
                "print(df_train_cleaned.columns.tolist())"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {
                "language": "python"
            },
            "source": [
                "# Log-transformering av utvalda variabler",
                "columns_to_transform = ['SalePrice', 'GrLivArea', 'LotArea']",
                "",
                "# Skapa nya log-transformerade kolumner",
                "for col in columns_to_transform:",
                "    if col in df_train_cleaned.columns:",
                "        df_train_cleaned[f'{col}_Log'] = np.log1p(df_train_cleaned[col])",
                "    if col in df_test_cleaned.columns:",
                "        df_test_cleaned[f'{col}_Log'] = np.log1p(df_test_cleaned[col])",
                "",
                "# Visa exempel på original och transformerade värden",
                "print(\"Exempel på original och transformerade värden (första 5 rader):\")",
                "cols_to_show = columns_to_transform + [f'{col}_Log' for col in columns_to_transform if col in df_train_cleaned.columns]",
                "display(df_train_cleaned[cols_to_show].head())"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {
                "language": "python"
            },
            "source": [
                "# One-hot encoding av kategoriska variabler",
                "def encode_categorical_variables(train_df, test_df):",
                "    # Identifiera kategoriska kolumner",
                "    categorical_columns = train_df.select_dtypes(include=['object']).columns",
                "    ",
                "    # Skapa one-hot encoding för träningsdata",
                "    train_encoded = pd.get_dummies(train_df, columns=categorical_columns)",
                "    ",
                "    # Skapa one-hot encoding för testdata, se till att vi får samma kolumner som i träningsdata",
                "    test_encoded = pd.get_dummies(test_df, columns=categorical_columns)",
                "    ",
                "    # Säkerställ att test har samma kolumner som träning",
                "    for col in train_encoded.columns:",
                "        if col not in test_encoded.columns:",
                "            test_encoded[col] = 0",
                "    ",
                "    # Ordna kolumnerna så de matchar",
                "    test_encoded = test_encoded[train_encoded.columns]",
                "    ",
                "    return train_encoded, test_encoded",
                "",
                "# Applicera encoding",
                "df_train_encoded, df_test_encoded = encode_categorical_variables(df_train_cleaned, df_test_cleaned)",
                "",
                "print(\"INFORMATION OM TRANSFORMATIONEN:\")",
                "print(f\"Ursprunglig träningsdata shape: {df_train_cleaned.shape}\")",
                "print(f\"Efter one-hot encoding shape: {df_train_encoded.shape}\")",
                "print(f\"\\nAntal nya kolumner skapade: {df_train_encoded.shape[1] - df_train_cleaned.shape[1]}\")",
                "print(\"\\nExempel på nya dummy-variabler (första 5):\")",
                "new_cols = set(df_train_encoded.columns) - set(df_train_cleaned.columns)",
                "print(list(new_cols)[:5])"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {
                "language": "python"
            },
            "source": [
                "# Skalning av numeriska variabler",
                "def scale_numeric_variables(train_df, test_df):",
                "    # Identifiera numeriska kolumner (exkludera tidigare skapade log-transformerade kolumner)",
                "    numeric_columns = train_df.select_dtypes(include=['int64', 'float64']).columns",
                "    numeric_columns = [col for col in numeric_columns if not col.endswith('_Log')]",
                "    ",
                "    # Initiera scaler",
                "    scaler = StandardScaler()",
                "    ",
                "    # Skapa kopior av dataframes för skalade värden",
                "    train_scaled = train_df.copy()",
                "    test_scaled = test_df.copy()",
                "    ",
                "    # Anpassa scaler på träningsdata och transformera både träning och test",
                "    train_scaled[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])",
                "    test_scaled[numeric_columns] = scaler.transform(test_df[numeric_columns])",
                "    ",
                "    return train_scaled, test_scaled, numeric_columns",
                "",
                "# Applicera skalning",
                "df_train_scaled, df_test_scaled, scaled_columns = scale_numeric_variables(df_train_encoded, df_test_encoded)",
                "",
                "print(\"INFORMATION OM SKALNINGEN:\")",
                "print(f\"Antal skalade variabler: {len(scaled_columns)}\")",
                "print(\"\\nExempel på original vs skalade värden (första 5 rader):\")",
                "example_cols = scaled_columns[:2]  # Visa bara första två variablerna för tydlighet",
                "display(df_train_scaled[example_cols].head())",
                "",
                "# Visa statistik för att verifiera skalningen",
                "print(\"\\nStatistik för skalade variabler (ska ha medelvärde ≈ 0 och std ≈ 1):\")",
                "scaled_stats = df_train_scaled[scaled_columns].describe().round(3)",
                "display(scaled_stats)",
                "",
                "# Spara de bearbetade datasetten",
                "df_train_scaled.to_csv('processed_train.csv', index=False)",
                "df_test_scaled.to_csv('processed_test.csv', index=False)",
                "print(\"\\nBearbetad data har sparats till 'processed_train.csv' och 'processed_test.csv'\")"
            ]
        }
    ]
}