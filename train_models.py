"""Train two small models from the supplied vgsales.csv dataset."""
from pathlib import Path
from zipfile import ZipFile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "vgsales.csv.zip"
DATA_PATH = ROOT / "vgsales.csv"
MODELS = ROOT / "models"

if ZIP_PATH.exists():
    with ZipFile(ZIP_PATH) as archive:
        archive.extract("vgsales.csv", ROOT)
if not DATA_PATH.exists():
    raise FileNotFoundError("Put vgsales.csv.zip beside this script and run it again.")

df = pd.read_csv(DATA_PATH).dropna(subset=["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"])

# Global_Sales is the sum of the four regional sales columns, so this is a
# transparent baseline demonstrating regression (rather than a real forecast).
reg_features = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(df[reg_features], df["Global_Sales"], test_size=0.2, random_state=42)
regression = LinearRegression().fit(Xr_train, yr_train)
r2 = r2_score(yr_test, regression.predict(Xr_test))

# Classification uses platform + year to predict the game genre.
genre_df = df.dropna(subset=["Genre", "Platform", "Year"])
Xc = genre_df[["Platform", "Year"]]
yc = genre_df["Genre"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
preprocess = ColumnTransformer([
    ("platform", OneHotEncoder(handle_unknown="ignore"), ["Platform"]),
    ("year", SimpleImputer(strategy="median"), ["Year"]),
])
classification = make_pipeline(preprocess, RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"))
classification.fit(Xc_train, yc_train)
accuracy = accuracy_score(yc_test, classification.predict(Xc_test))

MODELS.mkdir(exist_ok=True)
joblib.dump(regression, MODELS / "regression.joblib")
joblib.dump(classification, MODELS / "classification.joblib")
print(f"Regression R²: {r2:.4f}")
print(f"Classification accuracy: {accuracy:.4f}")
print(f"Saved models in {MODELS}")
