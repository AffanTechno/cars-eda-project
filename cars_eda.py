"""
AIML Mini Project 1: Exploratory Data Analysis (EDA) on Cars
Student: [Write your name]
Dataset file required in the same folder: Car details v3.csv

Run:
    pip install pandas numpy matplotlib
    python StudentName_Cars_EDA_Project.py

The script checks data quality, cleans selected fields, calculates statistics,
creates charts, and prints evidence-based observations. It does not train an ML model.
"""
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = Path("Car details v3.csv")
OUTPUT_DIR = Path("cars_eda_charts")
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Place 'Car details v3.csv' in the same folder as this Python file."
    )

raw = pd.read_csv(DATA_FILE)
df = raw.copy()
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

print("=" * 65)
print("1. DATASET OVERVIEW")
print("=" * 65)
print("Original shape:", raw.shape)
print("Columns:", list(df.columns))
print("\\nData types:\\n", df.dtypes)
print("\\nFirst five rows:\\n", df.head())
print("\\nNumerical summary:\\n", df.describe(include="number").T)
print("\\nCategorical summary:\\n", df.describe(include="object").T)

print("\\n" + "=" * 65)
print("2. DATA QUALITY CHECKS")
print("=" * 65)
print("Missing values before cleaning:\\n", df.isna().sum())
duplicate_count = int(df.duplicated().sum())
print("Duplicate rows:", duplicate_count)
df = df.drop_duplicates().copy()
print("Shape after removing exact duplicate rows:", df.shape)

for col in ["name", "fuel", "seller_type", "transmission", "owner"]:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()

for col in ["fuel", "seller_type", "transmission", "owner"]:
    if col in df.columns:
        df[col] = df[col].str.replace(r"\\s+", " ", regex=True).str.title()


def first_number(value):
    """Extract the first numeric value from text."""
    if pd.isna(value):
        return np.nan

    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


# Preserve original text columns and create numeric analysis columns.
for col in ["mileage", "engine", "max_power"]:
    if col in df.columns:
        df[col + "_numeric"] = df[col].map(first_number)

# Mark clearly invalid values as missing rather than guessing their correct value.
if "year" in df.columns:
    df.loc[(df["year"] < 1950) | (df["year"] > 2026), "year"] = np.nan
for col in ["selling_price", "km_driven", "seats", "mileage_numeric",
            "engine_numeric", "max_power_numeric"]:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

# Treat zero mileage and zero maximum power as missing values.
df.loc[df["mileage_numeric"] <= 0, "mileage_numeric"] = np.nan
df.loc[df["max_power_numeric"] <= 0, "max_power_numeric"] = np.nan

numeric_cols = ["year", "selling_price", "km_driven", "seats",
                "mileage_numeric", "engine_numeric", "max_power_numeric"]
for col in numeric_cols:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

for col in ["fuel", "seller_type", "transmission", "owner"]:
    if col in df.columns and df[col].isna().any():
        modes = df[col].mode(dropna=True)
        if not modes.empty:
            df[col] = df[col].fillna(modes.iloc[0])

print("\\nMissing values after cleaning:\\n", df.isna().sum())
print("Cleaned shape:", df.shape)
print("Note: median/mode filling is used only for analysis; original text columns are retained.")
print("Outliers are flagged using the IQR rule, not automatically deleted.")

print("\\n" + "=" * 65)
print("3. DESCRIPTIVE ANALYSIS")
print("=" * 65)
print(df.describe(include="all").T)
brand = df["name"].astype(str).str.split().str[0].str.title()
print("\\nMost frequent brand label (first word of car name):")
print(brand.value_counts().head(10))
print("\\nFuel counts:\\n", df["fuel"].value_counts())
print("\\nAverage price by fuel type:\\n",
      df.groupby("fuel")["selling_price"].agg(["count", "mean", "median"]).sort_values("mean", ascending=False))
print("\\nAverage price by transmission:\\n",
      df.groupby("transmission")["selling_price"].agg(["count", "mean", "median"]).sort_values("mean", ascending=False))

print("\\n" + "=" * 65)
print("4. OUTLIER CHECK (IQR RULE)")
print("=" * 65)
for col in ["selling_price", "km_driven", "mileage_numeric", "engine_numeric", "max_power_numeric"]:
    series = df[col].dropna()
    q1, q3 = series.quantile(.25), series.quantile(.75)
    iqr = q3 - q1
    count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
    print(f"{col}: {count} potential outliers; retained for analysis.")

# Plot a sample for scatter plots so charts remain readable.
plot_df = df.sample(min(1500, len(df)), random_state=42)
plt.rcParams.update({"figure.figsize": (8, 5), "font.size": 9})

def save_chart(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()

plt.figure()
plt.hist(df["selling_price"] / 100000, bins=35, edgecolor="black")
plt.title("Distribution of Car Selling Prices")
plt.xlabel("Selling price (₹ lakh)")
plt.ylabel("Number of cars")
save_chart("01_price_distribution.png")

plt.figure()
fuel_counts = df["fuel"].value_counts()
plt.bar(fuel_counts.index.astype(str), fuel_counts.values)
plt.title("Cars by Fuel Type")
plt.xlabel("Fuel type")
plt.ylabel("Number of records")
plt.xticks(rotation=20, ha="right")
save_chart("02_fuel_counts.png")

plt.figure()
fuel_order = df.groupby("fuel")["selling_price"].median().sort_values(ascending=False).index
box_data = [df.loc[df["fuel"] == fuel, "selling_price"] / 100000 for fuel in fuel_order]
plt.boxplot(box_data, tick_labels=fuel_order, showfliers=False)
plt.title("Selling Price by Fuel Type")
plt.xlabel("Fuel type")
plt.ylabel("Selling price (₹ lakh)")
plt.xticks(rotation=20, ha="right")
save_chart("03_price_fuel_boxplot.png")

plt.figure()
plt.scatter(plot_df["year"], plot_df["selling_price"] / 100000, alpha=.35, s=12)
plt.title("Model Year and Selling Price")
plt.xlabel("Manufacturing year")
plt.ylabel("Selling price (₹ lakh)")
save_chart("04_year_price_scatter.png")

plt.figure()
plt.scatter(plot_df["km_driven"], plot_df["selling_price"] / 100000, alpha=.35, s=12)
plt.title("Kilometres Driven and Selling Price")
plt.xlabel("Kilometres driven")
plt.ylabel("Selling price (₹ lakh)")
save_chart("05_km_price_scatter.png")

corr_cols = ["selling_price", "year", "km_driven", "mileage_numeric",
             "engine_numeric", "max_power_numeric", "seats"]
corr = df[corr_cols].corr(numeric_only=True)
plt.figure(figsize=(8, 6))
image = plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(image, label="Correlation coefficient")
labels = [c.replace("_numeric", "").replace("_", " ") for c in corr.columns]
plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.yticks(range(len(labels)), labels)
plt.title("Correlation Between Numerical Features")
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
save_chart("06_correlation_heatmap.png")

plt.figure()
top_brands = brand.value_counts().head(10).sort_values()
plt.barh(top_brands.index, top_brands.values)
plt.title("Top 10 Brands by Number of Listings")
plt.xlabel("Number of records")
plt.ylabel("Brand (first word of car name)")
save_chart("07_top_brands.png")

plt.figure()
plt.scatter(df["max_power_numeric"], df["selling_price"] / 100000, alpha=.3, s=12)
plt.title("Maximum Power and Selling Price")
plt.xlabel("Maximum power (numeric value extracted)")
plt.ylabel("Selling price (₹ lakh)")
save_chart("08_power_price.png")

print("\\nCharts saved to:", OUTPUT_DIR.resolve())
print("Created 8 charts. Review the charts and write observations supported by them.")
print("\\n" + "=" * 65)
print("5. KEY RELATIONSHIPS TO DISCUSS")
print("=" * 65)
print("Price-year correlation:", round(df["selling_price"].corr(df["year"]), 3))
print("Price-kilometres correlation:", round(df["selling_price"].corr(df["km_driven"]), 3))
print("Price-maximum-power correlation:", round(df["selling_price"].corr(df["max_power_numeric"]), 3))
print("Correlations show association, not proof of cause and effect.")
