import os
import io
import sqlite3
import hashlib
import secrets
from datetime import datetime

import numpy as np
import pandas as pd

from flask import (
    Flask,
    request,
    jsonify,
    render_template_string,
    send_file,
    redirect,
    session
)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

app = Flask(__name__)
app.secret_key = "FraudGuard_" + secrets.token_hex(16)

DATASET = "fraud transaction.csv"
DATABASE = "fraud_detection.db"
UPLOAD_FOLDER = "uploads"
ACTIVE_DATASET = os.path.join(UPLOAD_FOLDER, "active_dataset.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

NUMERIC_FEATURES = [
    "Amount",
    "Hour",
    "Frequency",
    "Customer_Age",
    "Previous_Transactions",
    "Average_Amount",
    "Distance_From_Usual_Location"
]

CATEGORICAL_FEATURES = [
    "Merchant_Category",
    "Location"
]

models = {}
model_results = {}
confusion_matrices = {}
roc_data = {}

dataset_df = pd.DataFrame()
dataset_scored_df = pd.DataFrame()
high_risk_df = pd.DataFrame()
primary_model = None


# =========================================================
# DATABASE
# =========================================================

def init_database():

    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            created_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_number TEXT,
            amount REAL,
            transaction_date TEXT,
            transaction_time TEXT,
            hour INTEGER,
            frequency REAL,
            merchant TEXT,
            location TEXT,
            age INTEGER,
            previous_transactions REAL,
            average_amount REAL,
            distance REAL,
            prediction TEXT,
            probability REAL,
            risk TEXT,
            factors TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def login_required():
    return "user_id" in session


init_database()


# =========================================================
# DEMO DATASET
# =========================================================

def create_demo_dataset():

    np.random.seed(42)

    n = 1000

    amount = np.random.exponential(1200, n)
    amount = np.clip(amount, 50, 15000)

    hour = np.random.randint(0, 24, n)

    frequency = np.random.randint(1, 25, n)

    age = np.random.randint(18, 70, n)

    previous_transactions = np.random.randint(
        1, 100, n
    )

    average_amount = np.random.uniform(
        300,
        2500,
        n
    )

    distance = np.random.uniform(
        0,
        150,
        n
    )

    merchant = np.random.choice(
        [
            "Retail",
            "Food",
            "Travel",
            "Online",
            "Entertainment",
            "Healthcare",
            "Electronics"
        ],
        n
    )

    location = np.random.choice(
        [
            "Coimbatore",
            "Chennai",
            "Bangalore",
            "Mumbai",
            "Delhi",
            "Hyderabad",
            "Madurai"
        ],
        n
    )

    score = (
        (amount > average_amount * 3).astype(int)
        + ((hour <= 4) | (hour >= 23)).astype(int)
        + (distance > 70).astype(int)
        + (frequency > 18).astype(int)
        + (amount > 5000).astype(int)
    )

    fraud = (score >= 2).astype(int)

    return pd.DataFrame({
        "Amount": amount,
        "Hour": hour,
        "Frequency": frequency,
        "Customer_Age": age,
        "Previous_Transactions": previous_transactions,
        "Average_Amount": average_amount,
        "Distance_From_Usual_Location": distance,
        "Merchant_Category": merchant,
        "Location": location,
        "Fraud": fraud
    })


# =========================================================
# COLUMN NORMALIZATION
# =========================================================

def normalize_columns(df):

    df = df.copy()

    aliases = {
        "amount": "Amount",
        "transaction_amount": "Amount",
        "transactionamount": "Amount",

        "hour": "Hour",
        "transaction_hour": "Hour",

        "frequency": "Frequency",
        "transaction_frequency": "Frequency",

        "customer_age": "Customer_Age",
        "age": "Customer_Age",

        "previous_transactions":
            "Previous_Transactions",

        "average_amount":
            "Average_Amount",

        "avg_amount":
            "Average_Amount",

        "distance":
            "Distance_From_Usual_Location",

        "distance_from_usual_location":
            "Distance_From_Usual_Location",

        "merchant":
            "Merchant_Category",

        "merchant_category":
            "Merchant_Category",

        "location":
            "Location",

        "fraud":
            "Fraud",

        "is_fraud":
            "Fraud",

        "fraudulent":
            "Fraud",

        "label":
            "Fraud",

        "fraud_label":
            "Fraud"
    }

    rename = {}

    for column in df.columns:

        cleaned = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if cleaned in aliases:
            rename[column] = aliases[cleaned]

    df.rename(
        columns=rename,
        inplace=True
    )

    return df


# =========================================================
# FLEXIBLE DATASET ADAPTER
# =========================================================

def prepare_uploaded_dataset(df):
    """Adapt a binary tabular fraud dataset to the app's common schema.

    The uploaded file may use different column names. The target is detected
    from common names such as Fraud, Class, Label, Target, Is_Fraud, etc.
    Missing fraud-model fields are automatically mapped from available numeric
    and categorical columns so different tabular fraud datasets can be used.
    """
    df = normalize_columns(df)

    if "Fraud" not in df.columns:
        candidates = []
        for col in df.columns:
            key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
            if key in {"class", "target", "label", "is_fraud", "fraud_flag",
                       "fraudulent", "fraud_status", "outcome", "result", "status"}:
                candidates.append(col)
        if candidates:
            df = df.rename(columns={candidates[0]: "Fraud"})

    if "Fraud" not in df.columns:
        raise ValueError(
            "The uploaded dataset must contain a binary target column such as Fraud, Class, Label, Target or Is_Fraud."
        )

    raw_target = df["Fraud"].copy()
    text = raw_target.astype(str).str.strip().str.lower()
    known = text.map({
        "fraud": 1, "fraudulent": 1, "yes": 1, "true": 1, "1": 1,
        "normal": 0, "legitimate": 0, "legit": 0, "no": 0, "false": 0, "0": 0
    })
    numeric = pd.to_numeric(raw_target, errors="coerce")
    target = known.fillna(numeric)

    if target.isna().any():
        uniques = list(pd.unique(raw_target.dropna()))
        if len(uniques) == 2:
            suspicious = [v for v in uniques if any(k in str(v).lower() for k in ["fraud", "yes", "true", "positive", "1"])]
            if suspicious:
                positive = suspicious[0]
                target = raw_target.apply(lambda v: 1 if str(v) == str(positive) else 0)
            else:
                mapping = {str(uniques[0]): 0, str(uniques[1]): 1}
                target = raw_target.astype(str).map(mapping)

    target = pd.to_numeric(target, errors="coerce")
    if target.isna().any() or len(target.dropna().unique()) != 2:
        raise ValueError("The target column must contain exactly two classes for Fraud/Normal prediction.")

    df["Fraud"] = target.astype(int)

    # Map known fields first. Then fill missing fields from available columns.
    numeric_available = [c for c in df.select_dtypes(include=np.number).columns if c != "Fraud"]
    used_numeric = set()
    for feature in NUMERIC_FEATURES:
        if feature in df.columns:
            used_numeric.add(feature)
            df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0)

    remaining_numeric = [c for c in numeric_available if c not in used_numeric]
    for feature in NUMERIC_FEATURES:
        if feature not in df.columns:
            source = remaining_numeric.pop(0) if remaining_numeric else None
            df[feature] = pd.to_numeric(df[source], errors="coerce").fillna(0) if source else 0

    categorical_available = [c for c in df.select_dtypes(include=["object", "category", "bool"]).columns if c != "Fraud"]
    used_cat = set()
    for feature in CATEGORICAL_FEATURES:
        if feature in df.columns:
            used_cat.add(feature)
            df[feature] = df[feature].fillna("Unknown").astype(str)

    remaining_cat = [c for c in categorical_available if c not in used_cat]
    for feature in CATEGORICAL_FEATURES:
        if feature not in df.columns:
            source = remaining_cat.pop(0) if remaining_cat else None
            df[feature] = df[source].fillna("Unknown").astype(str) if source else "Unknown"

    return df


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(path=None):

    global dataset_df, dataset_scored_df, models, model_results
    global confusion_matrices, roc_data, primary_model

    try:

        if path is None:
            path = ACTIVE_DATASET if os.path.exists(ACTIVE_DATASET) else None

        if path is None or not os.path.exists(path):
            dataset_df = pd.DataFrame()
            dataset_scored_df = pd.DataFrame()
            models = {}
            model_results = {}
            confusion_matrices = {}
            roc_data = {}
            primary_model = None
            return dataset_df

        else:

            extension = os.path.splitext(
                path
            )[1].lower()

            if extension == ".xlsx":

                df = pd.read_excel(path)

            else:

                df = pd.read_csv(
                    path,
                    on_bad_lines="skip"
                )

            df = prepare_uploaded_dataset(df)

            dataset_df = df

        train_models(dataset_df)

        return dataset_df

    except Exception as e:

        print(
            "Dataset loading error:",
            e
        )

        dataset_df = pd.DataFrame()
        dataset_scored_df = pd.DataFrame()
        models = {}
        model_results = {}
        confusion_matrices = {}
        roc_data = {}
        primary_model = None

        return dataset_df


# =========================================================
# RISK
# =========================================================

def calculate_risk(probability):

    if probability >= 0.50:
        return "High Risk"

    if probability >= 0.40:
        return "Medium Risk"

    return "Low Risk"


def get_risk_factors(row, probability):
    factors = []

    if probability >= 0.50:
        if row["Amount"] > row["Average_Amount"] * 3:
            factors.append("Transaction amount is unusually high.")
        if row["Hour"] <= 4 or row["Hour"] >= 23:
            factors.append("Transaction occurred during unusual hours.")
        if row["Distance_From_Usual_Location"] > 70:
            factors.append("Transaction location is far from usual location.")
        if row["Frequency"] > 18:
            factors.append("Transaction frequency is unusually high.")
        if row["Amount"] > 5000:
            factors.append("Large transaction amount detected.")

    if not factors:
        factors.append("No major suspicious factors detected.")

    return factors


# =========================================================
# TRAIN ML MODELS
# =========================================================

def train_models(df=None):

    global models
    global model_results
    global confusion_matrices
    global roc_data
    global primary_model
    global high_risk_df

    if df is None:
        df = dataset_df

    if df.empty:
        return

    X = df[
        NUMERIC_FEATURES +
        CATEGORICAL_FEATURES
    ].copy()

    y = df["Fraud"].astype(int)

    if len(y.unique()) < 2:

        print(
            "Dataset must contain both fraud and normal records."
        )

        return

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    algorithms = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced"
            )
    }

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

    except:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )
        )

    models = {}
    model_results = {}
    confusion_matrices = {}
    roc_data = {}

    for name, algorithm in algorithms.items():

        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                algorithm
            )
        ])

        try:

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            probabilities = (
                pipeline.predict_proba(
                    X_test
                )[:, 1]
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            try:

                auc = roc_auc_score(
                    y_test,
                    probabilities
                )

            except:

                auc = 0

            cm = confusion_matrix(
                y_test,
                predictions
            )

            try:

                fpr, tpr, _ = roc_curve(
                    y_test,
                    probabilities
                )

                roc_data[name] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist()
                }

            except:

                roc_data[name] = {
                    "fpr": [],
                    "tpr": []
                }

            models[name] = pipeline

            model_results[name] = {

                "accuracy":
                    round(
                        accuracy * 100,
                        2
                    ),

                "precision":
                    round(
                        precision * 100,
                        2
                    ),

                "recall":
                    round(
                        recall * 100,
                        2
                    ),

                "f1":
                    round(
                        f1 * 100,
                        2
                    ),

                "auc":
                    round(
                        auc * 100,
                        2
                    )
            }

            confusion_matrices[name] = (
                cm.tolist()
            )

        except Exception as e:

            print(
                "Model error:",
                name,
                e
            )

    if "Random Forest" in models:

        primary_model = models[
            "Random Forest"
        ]

    elif len(models) > 0:

        primary_model = list(
            models.values()
        )[0]

    else:

        primary_model = None

    high_risk_df = pd.DataFrame()

    # Score every dataset record so the Records module
    # can display High / Medium / Low Risk records.
    score_dataset_records()


def score_dataset_records():
    global dataset_scored_df

    dataset_scored_df = pd.DataFrame()

    if dataset_df.empty or primary_model is None:
        return

    try:
        x = dataset_df[
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ].copy()

        probabilities = primary_model.predict_proba(x)[:, 1]

        scored = dataset_df.copy()
        scored.insert(0, "Dataset_ID", range(1, len(scored) + 1))
        scored["Prediction"] = np.where(
            probabilities >= 0.50,
            "Fraud",
            "Normal"
        )
        scored["Probability"] = np.round(
            probabilities * 100,
            2
        )
        scored["Risk"] = [
            calculate_risk(float(p))
            for p in probabilities
        ]
        scored["Risk_Factors"] = [
            ", ".join(
                get_risk_factors(
                    row,
                    float(p)
                )
            )
            for (_, row), p in zip(
                dataset_df.iterrows(),
                probabilities
            )
        ]

        dataset_scored_df = scored

    except Exception as e:
        print("Dataset scoring error:", e)


# =========================================================
# PREDICTION
# =========================================================

def predict_transaction(data):

    if primary_model is None:

        raise Exception(
            "ML model is not trained."
        )

    row = {

        "Amount":
            float(
                data.get(
                    "amount",
                    0
                )
            ),

        "Hour":
            int(
                data.get(
                    "hour",
                    12
                )
            ),

        "Frequency":
            float(
                data.get(
                    "frequency",
                    1
                )
            ),

        "Customer_Age":
            float(
                data.get(
                    "age",
                    30
                )
            ),

        "Previous_Transactions":
            float(
                data.get(
                    "previous_transactions",
                    1
                )
            ),

        "Average_Amount":
            float(
                data.get(
                    "average_amount",
                    1000
                )
            ),

        "Distance_From_Usual_Location":
            float(
                data.get(
                    "distance",
                    0
                )
            ),

        "Merchant_Category":
            str(
                data.get(
                    "merchant",
                    "Unknown"
                )
            ),

        "Location":
            str(
                data.get(
                    "location",
                    "Unknown"
                )
            )
    }

    input_df = pd.DataFrame([row])

    probability = float(
        primary_model.predict_proba(
            input_df
        )[0][1]
    )

    prediction = (
        "Fraud"
        if probability >= 0.50
        else "Normal"
    )

    risk = calculate_risk(
        probability
    )

    factors = get_risk_factors(
        row,
        probability
    )

    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk": risk,
        "factors": factors
    }
    
def save_prediction(data, result):

    global high_risk_df

    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        INSERT INTO predictions (
            account_number,
            amount,
            transaction_date,
            transaction_time,
            hour,
            frequency,
            merchant,
            location,
            age,
            previous_transactions,
            average_amount,
            distance,
            prediction,
            probability,
            risk,
            factors,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        data.get("account_number", ""),

        float(data.get("amount", 0)),

        data.get("transaction_date", ""),

        data.get("transaction_time", ""),

        int(data.get("hour", 12)),

        float(data.get("frequency", 1)),

        data.get("merchant", ""),

        data.get("location", ""),

        int(data.get("age", 30)),

        float(data.get("previous_transactions", 1)),

        float(data.get("average_amount", 1000)),

        float(data.get("distance", 0)),

        result["prediction"],

        result["probability"],

        result["risk"],

        ", ".join(result["factors"]),

        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    ))

    conn.commit()
    conn.close()


    # =====================================================
    # ADD ONLY HIGH-RISK MANUAL PREDICTIONS
    # =====================================================

    if result["risk"] == "High Risk":

        new_record = pd.DataFrame([{

            "Account_Number":
                data.get(
                    "account_number",
                    ""
                ),

            "Amount":
                float(
                    data.get(
                        "amount",
                        0
                    )
                ),

            "Transaction_Date":
                data.get(
                    "transaction_date",
                    ""
                ),

            "Transaction_Time":
                data.get(
                    "transaction_time",
                    ""
                ),

            "Hour":
                int(
                    data.get(
                        "hour",
                        12
                    )
                ),

            "Frequency":
                float(
                    data.get(
                        "frequency",
                        1
                    )
                ),

            "Merchant_Category":
                data.get(
                    "merchant",
                    ""
                ),

            "Location":
                data.get(
                    "location",
                    ""
                ),

            "Customer_Age":
                int(
                    data.get(
                        "age",
                        30
                    )
                ),

            "Previous_Transactions":
                float(
                    data.get(
                        "previous_transactions",
                        1
                    )
                ),

            "Average_Amount":
                float(
                    data.get(
                        "average_amount",
                        1000
                    )
                ),

            "Distance_From_Usual_Location":
                float(
                    data.get(
                        "distance",
                        0
                    )
                ),

            "Prediction":
                result["prediction"],

            "Fraud_Probability":
                result["probability"] / 100,

            "Risk":
                result["risk"],

            "Risk_Factors":
                ", ".join(
                    result["factors"]
                ),

            "Created_At":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

        }])

        high_risk_df = pd.concat(
            [
                high_risk_df,
                new_record
            ],
            ignore_index=True
        )


# =========================================================
# LOGIN
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        conn = sqlite3.connect(
            DATABASE
        )

        user = conn.execute("""
            SELECT id, username, password
            FROM users
            WHERE username = ?
        """, (
            username,
        )).fetchone()

        conn.close()

        if (
            user
            and
            user[2] ==
            hash_password(password)
        ):

            session["user_id"] = user[0]
            session["username"] = user[1]

            return redirect("/")

        return render_template_string(
            LOGIN_HTML,
            error="Invalid username or password."
        )

    return render_template_string(
        LOGIN_HTML,
        error=""
    )


# =========================================================
# SIGNUP
# =========================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if (
            not username
            or
            not email
            or
            not password
        ):

            return render_template_string(
                SIGNUP_HTML,
                error="Please fill all fields."
            )

        try:

            conn = sqlite3.connect(
                DATABASE
            )

            conn.execute("""
                INSERT INTO users
                (username, email, password, created_at)
                VALUES (?, ?, ?, ?)
            """, (

                username,

                email,

                hash_password(
                    password
                ),

                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ))

            conn.commit()
            conn.close()

            return redirect(
                "/login"
            )

        except sqlite3.IntegrityError:

            return render_template_string(
                SIGNUP_HTML,
                error="Username or email already exists."
            )

    return render_template_string(
        SIGNUP_HTML,
        error=""
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        "/login"
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    if not login_required():

        return redirect(
            "/login"
        )

    return render_template_string(
        HTML_PAGE,
        username=session.get(
            "username",
            "User"
        )
    )


# =========================================================
# MANUAL PREDICTION DATA HELPERS
# =========================================================

def get_manual_predictions():
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )
    conn.close()
    return df


def get_manual_high_risk():
    df = get_manual_predictions()
    if df.empty:
        return df

    return df[
        df["risk"].astype(str).str.strip() == "High Risk"
    ].copy()


# =========================================================
# DASHBOARD APIs
# =========================================================

def _group_counts(df, column):
    if df.empty or column not in df.columns:
        return {"labels": [], "values": []}

    grouped = (
        df[column]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    return {
        "labels": grouped.index.tolist(),
        "values": [int(v) for v in grouped.values]
    }


def _fraud_group_counts(df, column):
    if df.empty or column not in df.columns:
        return {"labels": [], "values": []}

    temp = df.copy()
    if "Fraud" in temp.columns:
        temp["_fraud"] = pd.to_numeric(
            temp["Fraud"], errors="coerce"
        ).fillna(0).astype(int).eq(1)
    elif "Prediction" in temp.columns:
        temp["_fraud"] = temp["Prediction"].astype(str).eq("Fraud")
    else:
        return {"labels": [], "values": []}

    grouped = (
        temp.groupby(column)["_fraud"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    return {
        "labels": [str(v) for v in grouped.index],
        "values": [int(v) for v in grouped.values]
    }


@app.route("/api/dashboard")
def dashboard_api():

    if not login_required():
        return jsonify({"error": "Please login."}), 401

    if dataset_df.empty:
        return jsonify({
            "records": 0,
            "fraud": 0,
            "normal": 0,
            "fraud_rate": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "charts": {}
        })

    scored = dataset_scored_df.copy()

    if scored.empty:
        fraud = int(dataset_df["Fraud"].sum())
        normal = len(dataset_df) - fraud
        return jsonify({
            "records": len(dataset_df),
            "fraud": fraud,
            "normal": normal,
            "fraud_rate": round(fraud / len(dataset_df) * 100, 2),
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "charts": {}
        })

    # Dashboard totals must represent the uploaded dataset itself,
    # not the model's predicted classes. This keeps Dataset Upload,
    # Dashboard and Dataset Analysis consistent.
    actual_fraud = pd.to_numeric(
        dataset_df["Fraud"], errors="coerce"
    ).fillna(0).astype(int)
    fraud = int((actual_fraud == 1).sum())
    normal = int((actual_fraud == 0).sum())
    total = len(dataset_df)

    risk_counts = (
        scored["Risk"]
        .value_counts()
        .to_dict()
    )

    hour_counts = _group_counts(scored, "Hour")
    merchant_counts = _fraud_group_counts(scored, "Merchant_Category")
    location_counts = _fraud_group_counts(scored, "Location")

    amount_bins = pd.cut(
        scored["Amount"],
        bins=[-float("inf"), 500, 1000, 5000, 10000, float("inf")],
        labels=["≤ ₹500", "₹501–₹1,000", "₹1,001–₹5,000", "₹5,001–₹10,000", "> ₹10,000"]
    )
    amount_counts = amount_bins.value_counts().sort_index()

    frequency_bins = pd.cut(
        scored["Frequency"],
        bins=[-float("inf"), 5, 10, 18, float("inf")],
        labels=["1–5", "6–10", "11–18", ">18"]
    )
    frequency_counts = frequency_bins.value_counts().sort_index()

    return jsonify({
        "records": total,
        "fraud": fraud,
        "normal": normal,
        "fraud_rate": round(fraud / total * 100, 2) if total else 0,
        "high_risk": int(risk_counts.get("High Risk", 0)),
        "medium_risk": int(risk_counts.get("Medium Risk", 0)),
        "low_risk": int(risk_counts.get("Low Risk", 0)),
        "charts": {
            "hour": hour_counts,
            "merchant": merchant_counts,
            "location": location_counts,
            "amount": {
                "labels": [str(v) for v in amount_counts.index],
                "values": [int(v) for v in amount_counts.values]
            },
            "frequency": {
                "labels": [str(v) for v in frequency_counts.index],
                "values": [int(v) for v in frequency_counts.values]
            },
            "risk": {
                "labels": ["High Risk", "Medium Risk", "Low Risk"],
                "values": [
                    int(risk_counts.get("High Risk", 0)),
                    int(risk_counts.get("Medium Risk", 0)),
                    int(risk_counts.get("Low Risk", 0))
                ]
            }
        }
    })


@app.route("/api/manual-dashboard")
def manual_dashboard_api():

    if not login_required():
        return jsonify({"error": "Please login."}), 401

    df = get_manual_predictions()

    if df.empty:
        return jsonify({
            "records": 0,
            "fraud": 0,
            "normal": 0,
            "fraud_rate": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "charts": {}
        })

    prediction_text = df["prediction"].astype(str).str.strip()
    fraud = int((prediction_text == "Fraud").sum())
    normal = int((prediction_text == "Normal").sum())
    total = len(df)

    risk_counts = (
        df["risk"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    temp = df.copy()
    temp["Prediction"] = temp["prediction"].astype(str)
    temp["Merchant_Category"] = temp["merchant"].astype(str)
    temp["Location"] = temp["location"].astype(str)
    temp["Hour"] = pd.to_numeric(temp["hour"], errors="coerce").fillna(0)

    amount_bins = pd.cut(
        pd.to_numeric(temp["amount"], errors="coerce").fillna(0),
        bins=[-float("inf"), 500, 1000, 5000, 10000, float("inf")],
        labels=["≤ ₹500", "₹501–₹1,000", "₹1,001–₹5,000", "₹5,001–₹10,000", "> ₹10,000"]
    )
    amount_counts = amount_bins.value_counts().sort_index()

    frequency_bins = pd.cut(
        pd.to_numeric(temp["frequency"], errors="coerce").fillna(0),
        bins=[-float("inf"), 5, 10, 18, float("inf")],
        labels=["1–5", "6–10", "11–18", ">18"]
    )
    frequency_counts = frequency_bins.value_counts().sort_index()

    return jsonify({
        "records": total,
        "fraud": fraud,
        "normal": normal,
        "fraud_rate": round(fraud / total * 100, 2) if total else 0,
        "high_risk": int(risk_counts.get("High Risk", 0)),
        "medium_risk": int(risk_counts.get("Medium Risk", 0)),
        "low_risk": int(risk_counts.get("Low Risk", 0)),
        "charts": {
            "hour": _group_counts(temp, "Hour"),
            "merchant": _fraud_group_counts(temp, "Merchant_Category"),
            "location": _fraud_group_counts(temp, "Location"),
            "amount": {
                "labels": [str(v) for v in amount_counts.index],
                "values": [int(v) for v in amount_counts.values]
            },
            "frequency": {
                "labels": [str(v) for v in frequency_counts.index],
                "values": [int(v) for v in frequency_counts.values]
            },
            "risk": {
                "labels": ["High Risk", "Medium Risk", "Low Risk"],
                "values": [
                    int(risk_counts.get("High Risk", 0)),
                    int(risk_counts.get("Medium Risk", 0)),
                    int(risk_counts.get("Low Risk", 0))
                ]
            }
        }
    })


# =========================================================
# MANUAL MODULE ANALYSIS API
# =========================================================

@app.route("/api/manual-analysis")
def manual_analysis_api():

    if not login_required():
        return jsonify({"error": "Please login."}), 401

    df = get_manual_predictions()

    if df.empty:
        return jsonify({
            "records": [],
            "columns": []
        })

    analysis = [
        {
            "Feature": "Manual Predictions",
            "Value": len(df)
        },
        {
            "Feature": "Fraud Predictions",
            "Value": int((df["prediction"] == "Fraud").sum())
        },
        {
            "Feature": "Normal Predictions",
            "Value": int((df["prediction"] == "Normal").sum())
        },
        {
            "Feature": "High Risk Records",
            "Value": int((df["risk"] == "High Risk").sum())
        },
        {
            "Feature": "Average Fraud Probability",
            "Value": round(float(df["probability"].mean()), 2)
        }
    ]

    return jsonify({
        "records": analysis,
        "columns": ["Feature", "Value"]
    })


# =========================================================
# DATASET API
# =========================================================

@app.route(
    "/api/dataset"
)
def dataset_api():

    if not login_required():

        return jsonify({
            "error": "Please login."
        }), 401

    if dataset_df.empty:

        return jsonify({
            "records": []
        })

    preview = (
        dataset_df
        .head(50)
        .replace({np.nan: ""})
        .to_dict(
            orient="records"
        )
    )

    return jsonify({

        "records":
            preview,

        "columns":
            list(
                dataset_df.columns
            )
    })


# =========================================================
# UPLOAD API
# =========================================================

@app.route(
    "/api/upload",
    methods=["POST"]
)
def upload_api():

    global dataset_df, dataset_scored_df

    if not login_required():
        return jsonify({"error": "Please login first."}), 401

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "Please select a CSV or XLSX file."}), 400

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in {".csv", ".xlsx"}:
        return jsonify({"error": "Only CSV and XLSX files are supported."}), 400

    try:
        raw = pd.read_excel(file) if extension == ".xlsx" else pd.read_csv(file, on_bad_lines="skip")
        if raw.empty:
            return jsonify({"error": "The uploaded dataset is empty."}), 400

        prepared = prepare_uploaded_dataset(raw)

        # Persist the latest uploaded dataset so it becomes the active dataset.
        prepared.to_csv(ACTIVE_DATASET, index=False)

        dataset_df = prepared.copy()
        dataset_scored_df = pd.DataFrame()
        train_models(dataset_df)

        if dataset_scored_df.empty:
            return jsonify({"error": "The dataset was loaded, but the ML model could not score the records."}), 400

        fraud_actual = int(dataset_df["Fraud"].sum())
        normal_actual = int(len(dataset_df) - fraud_actual)

        return jsonify({
            "message": "Dataset uploaded, adapted and trained successfully.",
            "records": int(len(dataset_df)),
            "fraud": fraud_actual,
            "normal": normal_actual,
            "columns": list(dataset_df.columns),
            "active_file": os.path.basename(ACTIVE_DATASET)
        })

    except Exception as e:
        dataset_df = pd.DataFrame()
        dataset_scored_df = pd.DataFrame()
        return jsonify({"error": str(e)}), 400


# =========================================================
# MODEL API
# =========================================================

@app.route(
    "/api/models"
)
def models_api():

    if not login_required():

        return jsonify({
            "error": "Please login."
        }), 401

    return jsonify({

        "models":
            model_results,

        "confusion_matrices":
            confusion_matrices,

        "roc":
            roc_data
    })


# =========================================================
# RISK RECORD APIs
# =========================================================

def risk_name(level):
    return {
        "high": "High Risk",
        "medium": "Medium Risk",
        "low": "Low Risk"
    }.get(level.lower())


def dataset_risk_records(level):
    if dataset_scored_df.empty:
        return []

    risk = risk_name(level)
    if not risk:
        return []

    df = dataset_scored_df[
        dataset_scored_df["Risk"] == risk
    ].copy()

    records = []

    for _, row in df.head(500).iterrows():
        records.append({
            "id": int(row["Dataset_ID"]),
            "source": "dataset",
            "account_number": str(row.get("Account_Number", row["Dataset_ID"])),
            "amount": float(row.get("Amount", 0)),
            "transaction_date": str(row.get("Transaction_Date", "")),
            "transaction_time": str(row.get("Transaction_Time", "")),
            "hour": int(float(row.get("Hour", 0))),
            "frequency": float(row.get("Frequency", 0)),
            "merchant": str(row.get("Merchant_Category", "")),
            "location": str(row.get("Location", "")),
            "age": int(float(row.get("Customer_Age", 0))),
            "previous_transactions": float(row.get("Previous_Transactions", 0)),
            "average_amount": float(row.get("Average_Amount", 0)),
            "distance": float(row.get("Distance_From_Usual_Location", 0)),
            "prediction": str(row.get("Prediction", "")),
            "probability": float(row.get("Probability", 0)),
            "risk": str(row.get("Risk", "")),
            "factors": str(row.get("Risk_Factors", "")),
            "actual_fraud": int(row.get("Fraud", 0))
        })

    return records


@app.route("/api/risk-summary/<source>")
def risk_summary(source):

    if not login_required():
        return jsonify({"error": "Please login."}), 401

    if source == "dataset":
        records = dataset_scored_df
        if records.empty:
            return jsonify({
                "High Risk": 0,
                "Medium Risk": 0,
                "Low Risk": 0
            })

        counts = records["Risk"].value_counts().to_dict()

    elif source == "manual":
        df = get_manual_predictions()
        if df.empty:
            counts = {}
        else:
            counts = df["risk"].value_counts().to_dict()

    else:
        return jsonify({"error": "Invalid source"}), 400

    return jsonify({
        "High Risk": int(counts.get("High Risk", 0)),
        "Medium Risk": int(counts.get("Medium Risk", 0)),
        "Low Risk": int(counts.get("Low Risk", 0))
    })


@app.route("/api/risk-records/<source>/<level>")
def risk_records(source, level):

    if not login_required():
        return jsonify({"error": "Please login."}), 401

    if not risk_name(level):
        return jsonify({"error": "Invalid risk level"}), 400

    if source == "dataset":
        records = dataset_risk_records(level)

    elif source == "manual":
        risk = risk_name(level)
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT *
            FROM predictions
            WHERE risk = ?
            ORDER BY id DESC
            LIMIT 500
        """, (risk,)).fetchall()
        conn.close()

        records = []
        for row in rows:
            records.append({
                "id": row["id"],
                "source": "manual",
                "account_number": row["account_number"],
                "amount": float(row["amount"] or 0),
                "transaction_date": row["transaction_date"],
                "transaction_time": row["transaction_time"],
                "hour": int(row["hour"] or 0),
                "frequency": float(row["frequency"] or 0),
                "merchant": row["merchant"],
                "location": row["location"],
                "age": int(row["age"] or 0),
                "previous_transactions": float(row["previous_transactions"] or 0),
                "average_amount": float(row["average_amount"] or 0),
                "distance": float(row["distance"] or 0),
                "prediction": row["prediction"],
                "probability": float(row["probability"] or 0),
                "risk": row["risk"],
                "factors": row["factors"],
                "created_at": row["created_at"]
            })

    else:
        return jsonify({"error": "Invalid source"}), 400

    return jsonify({
        "risk": risk_name(level),
        "count": len(records),
        "records": records
    })


@app.route("/api/dataset-analysis")
def dataset_analysis_api():

    if not login_required():
        return jsonify({"error": "Please login."}), 401

    if dataset_df.empty:
        return jsonify({"error": "No dataset loaded"}), 400

    df = dataset_df.copy()
    total = len(df)
    fraud_count = int(df["Fraud"].sum())
    normal_count = total - fraud_count

    numeric_stats = {}
    for column in df.select_dtypes(include="number").columns:
        numeric_stats[column] = {
            "mean": round(float(df[column].mean()), 2),
            "min": round(float(df[column].min()), 2),
            "max": round(float(df[column].max()), 2)
        }

    return jsonify({
        "total_records": total,
        "fraud_records": fraud_count,
        "normal_records": normal_count,
        "fraud_percentage": round(
            fraud_count / total * 100, 2
        ) if total else 0,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_statistics": numeric_stats,
        "columns": list(df.columns)
    })


# =========================================================
# HIGH RISK API
# =========================================================

@app.route(
    "/api/high-risk"
)
def high_risk_api():

    if not login_required():

        return jsonify({
            "error": "Please login."
        }), 401

    records = get_manual_high_risk()

    if records.empty:
        return jsonify({
            "records": []
        })

    output = []

    for _, row in records.head(100).iterrows():
        output.append({
            "Account_Number": row.get("account_number", ""),
            "Amount": float(row.get("amount", 0)),
            "Transaction_Date": row.get("transaction_date", ""),
            "Transaction_Time": row.get("transaction_time", ""),
            "Hour": int(row.get("hour", 0)),
            "Frequency": float(row.get("frequency", 0)),
            "Merchant_Category": row.get("merchant", ""),
            "Location": row.get("location", ""),
            "Customer_Age": int(row.get("age", 0)),
            "Previous_Transactions": float(row.get("previous_transactions", 0)),
            "Average_Amount": float(row.get("average_amount", 0)),
            "Distance_From_Usual_Location": float(row.get("distance", 0)),
            "Prediction": row.get("prediction", ""),
            "Fraud_Probability": float(row.get("probability", 0)),
            "Risk": row.get("risk", ""),
            "Risk_Factors": row.get("factors", ""),
            "Created_At": row.get("created_at", "")
        })

    return jsonify({
        "records": output
    })


# =========================================================
# PREDICTION API
# =========================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def prediction_api():

    if not login_required():

        return jsonify({
            "error": "Please login."
        }), 401

    try:

        data = request.get_json()

        result = predict_transaction(
            data
        )

        save_prediction(
            data,
            result
        )

        return jsonify(
            result
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# =========================================================
# HISTORY API
# =========================================================

@app.route(
    "/api/history"
)
def history_api():

    if not login_required():

        return jsonify({
            "error": "Please login."
        }), 401

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
        LIMIT 100
    """).fetchall()

    conn.close()

    return jsonify({

        "records": [
            dict(row)
            for row in rows
        ]

    })


# =========================================================
# DOWNLOAD HIGH RISK
# =========================================================

@app.route(
    "/api/high-risk/download"
)
def download_high_risk():

    if not login_required():
        return redirect("/login")

    records = get_manual_high_risk()

    if records.empty:
        df = pd.DataFrame(
            columns=[
                "Account_Number", "Amount", "Transaction_Date",
                "Transaction_Time", "Merchant_Category", "Location",
                "Prediction", "Fraud_Probability", "Risk",
                "Risk_Factors", "Created_At"
            ]
        )
    else:
        df = pd.DataFrame({
            "Account_Number": records["account_number"],
            "Amount": records["amount"],
            "Transaction_Date": records["transaction_date"],
            "Transaction_Time": records["transaction_time"],
            "Merchant_Category": records["merchant"],
            "Location": records["location"],
            "Prediction": records["prediction"],
            "Fraud_Probability": records["probability"],
            "Risk": records["risk"],
            "Risk_Factors": records["factors"],
            "Created_At": records["created_at"]
        })

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="High Risk"
        )

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="high_risk_transactions.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =========================================================
# DOWNLOAD HIGH / MEDIUM / LOW RISK RECORDS
# =========================================================

@app.route("/api/risk-download/<level>")
def download_risk_records(level):

    if not login_required():
        return redirect("/login")

    risk = risk_name(level)

    if not risk:
        return jsonify({"error": "Invalid risk level"}), 400

    source = request.args.get("source", "dataset")

    try:

        if source == "dataset":

            if dataset_scored_df.empty:
                df = pd.DataFrame()
            else:
                df = dataset_scored_df[
                    dataset_scored_df["Risk"] == risk
                ].copy()

            if not df.empty:
                df.rename(columns={
                    "Dataset_ID": "Record_ID",
                    "Probability": "Fraud_Probability",
                    "Risk_Factors": "Risk_Factors"
                }, inplace=True)

        elif source == "manual":

            conn = sqlite3.connect(DATABASE)

            df = pd.read_sql_query(
                """
                SELECT
                    id AS Record_ID,
                    account_number AS Account_Number,
                    amount AS Amount,
                    transaction_date AS Transaction_Date,
                    transaction_time AS Transaction_Time,
                    hour AS Hour,
                    frequency AS Frequency,
                    merchant AS Merchant_Category,
                    location AS Location,
                    age AS Customer_Age,
                    previous_transactions AS Previous_Transactions,
                    average_amount AS Average_Amount,
                    distance AS Distance_From_Usual_Location,
                    prediction AS Prediction,
                    probability AS Fraud_Probability,
                    risk AS Risk,
                    factors AS Risk_Factors,
                    created_at AS Created_At
                FROM predictions
                WHERE risk = ?
                ORDER BY id DESC
                """,
                conn,
                params=(risk,)
            )

            conn.close()

        else:
            return jsonify({"error": "Invalid source"}), 400

        if df.empty:
            df = pd.DataFrame(columns=[
                "Record_ID",
                "Account_Number",
                "Amount",
                "Transaction_Date",
                "Transaction_Time",
                "Hour",
                "Frequency",
                "Merchant_Category",
                "Location",
                "Customer_Age",
                "Previous_Transactions",
                "Average_Amount",
                "Distance_From_Usual_Location",
                "Prediction",
                "Fraud_Probability",
                "Risk",
                "Risk_Factors",
                "Created_At"
            ])

        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            df.to_excel(
                writer,
                index=False,
                sheet_name=risk.replace(" ", "_")
            )

        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name=f"{level.lower()}_risk_transactions.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# =========================================================
# DOWNLOAD DATASET REPORT
# =========================================================

@app.route(
    "/api/dataset-report"
)
def dataset_report():

    if not login_required():

        return redirect(
            "/login"
        )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        dataset_df.to_excel(
            writer,
            index=False,
            sheet_name="Dataset"
        )

        pd.DataFrame(
            model_results
        ).T.to_excel(
            writer,
            sheet_name="Model Results"
        )

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="fraud_dataset_report.xlsx",
        mimetype=
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =========================================================
# DOWNLOAD PREDICTION HISTORY
# =========================================================

@app.route(
    "/api/manual/download"
)
def download_history():

    if not login_required():

        return redirect(
            "/login"
        )

    conn = sqlite3.connect(
        DATABASE
    )

    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )

    conn.close()

    if not df.empty:

        duplicate_columns = [

            "account_number",
            "amount",
            "transaction_date",
            "transaction_time",
            "hour",
            "frequency",
            "merchant",
            "location",
            "age",
            "previous_transactions",
            "average_amount",
            "distance",
            "prediction",
            "probability",
            "risk",
            "factors"

        ]

        available_columns = [
            column
            for column in duplicate_columns
            if column in df.columns
        ]

        if available_columns:

            df = df.drop_duplicates(
                subset=available_columns,
                keep="first"
            )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Predictions"
        )

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="prediction_history.xlsx",
        mimetype=
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =========================================================
# LOGIN HTML
# =========================================================

LOGIN_HTML = """
<!DOCTYPE html>
<html>

<head>

<title>FraudGuard AI - Login</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    font-family: Arial, sans-serif;
    background: linear-gradient(
        135deg,
        #080d22,
        #18245c,
        #32105f
    );
    color: white;
}

.box {
    width: 390px;
    padding: 40px;
    border-radius: 25px;
    background: rgba(255,255,255,.09);
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 60px rgba(0,0,0,.4);
}

h1 {
    text-align: center;
}

input {
    width: 100%;
    padding: 14px;
    margin: 10px 0;
    border: none;
    border-radius: 10px;
}

button {
    width: 100%;
    padding: 14px;
    margin-top: 15px;
    border: none;
    border-radius: 10px;
    background: #7657ff;
    color: white;
    font-weight: bold;
    cursor: pointer;
}

a {
    color: #b9aaff;
}

.error {
    color: #ff8c8c;
    text-align: center;
}

</style>

</head>

<body>

<div class="box">

<h1>🛡️ FraudGuard AI</h1>

<p style="text-align:center;">
Fraud Transaction Detection
</p>

{% if error %}
<p class="error">{{ error }}</p>
{% endif %}

<form method="POST">

<input
    name="username"
    placeholder="Username"
    required
>

<input
    type="password"
    name="password"
    placeholder="Password"
    required
>

<button type="submit">
LOGIN
</button>

</form>

<p style="text-align:center;">
Don't have an account?
<a href="/signup">Create Account</a>
</p>

</div>

</body>
</html>
"""


# =========================================================
# SIGNUP HTML
# =========================================================

SIGNUP_HTML = """
<!DOCTYPE html>
<html>

<head>

<title>FraudGuard AI - Signup</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    font-family: Arial, sans-serif;
    background: linear-gradient(
        135deg,
        #080d22,
        #18245c,
        #32105f
    );
    color: white;
}

.box {
    width: 390px;
    padding: 40px;
    border-radius: 25px;
    background: rgba(255,255,255,.09);
    backdrop-filter: blur(20px);
}

h1 {
    text-align: center;
}

input {
    width: 100%;
    padding: 14px;
    margin: 10px 0;
    border: none;
    border-radius: 10px;
}

button {
    width: 100%;
    padding: 14px;
    margin-top: 15px;
    border: none;
    border-radius: 10px;
    background: #7657ff;
    color: white;
    font-weight: bold;
    cursor: pointer;
}

a {
    color: #b9aaff;
}

.error {
    color: #ff8c8c;
    text-align: center;
}

</style>

</head>

<body>

<div class="box">

<h1>🛡️ Create Account</h1>

{% if error %}
<p class="error">{{ error }}</p>
{% endif %}

<form method="POST">

<input
    name="username"
    placeholder="Username"
    required
>

<input
    type="email"
    name="email"
    placeholder="Email"
    required
>

<input
    type="password"
    name="password"
    placeholder="Password"
    required
>

<button type="submit">
CREATE ACCOUNT
</button>

</form>

<p style="text-align:center;">
Already have an account?
<a href="/login">Login</a>
</p>

</div>

</body>
</html>
"""


# =========================================================
# MAIN WEB APPLICATION
# =========================================================

HTML_PAGE = r"""
<!DOCTYPE html>
<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>FraudGuard AI</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #070b1c,
            #10183b,
            #241044
        );

    color: white;

    min-height: 100vh;
}


/* SIDEBAR */

.sidebar {

    position: fixed;

    left: 0;
    top: 0;

    width: 260px;

    height: 100vh;

    background:
        rgba(10,15,40,.96);

    border-right:
        1px solid
        rgba(255,255,255,.1);

    padding: 25px 18px;

    z-index: 1000;
}

.logo {

    text-align: center;

    font-size: 25px;

    font-weight: bold;

    margin-bottom: 35px;
}

.logo span {

    color: #8b70ff;
}

.nav {

    display: flex;

    flex-direction: column;

    gap: 9px;
}

.nav button {

    width: 100%;

    padding: 14px;

    border: none;

    border-radius: 12px;

    background:
        transparent;

    color:
        #b8bdd5;

    text-align: left;

    font-size: 14px;

    cursor: pointer;

    transition:
        .25s;
}

.nav button:hover {

    background:
        rgba(118,87,255,.2);

    color: white;

    transform:
        translateX(3px);
}

.nav button.active {

    background:
        linear-gradient(
            135deg,
            #6954ff,
            #9a52e8
        );

    color: white;

    box-shadow:
        0 8px 25px
        rgba(100,80,255,.25);
}

.logout {

    position: absolute;

    bottom: 25px;

    left: 18px;

    right: 18px;

    text-decoration: none;

    display: block;

    text-align: center;

    padding: 13px;

    border-radius: 12px;

    background:
        rgba(255,70,90,.12);

    color:
        #ff8795;
}


/* MAIN */

.main {

    margin-left: 260px;

    padding: 30px;

    min-height: 100vh;
}

.topbar {

    display: flex;

    justify-content:
        space-between;

    align-items:
        center;

    margin-bottom: 30px;
}

.topbar h1 {

    margin: 0;

    font-size: 30px;
}

.user {

    background:
        rgba(255,255,255,.08);

    padding: 10px 18px;

    border-radius: 20px;
}


/* PAGES */

.page {

    display: none;
}

.page.active {

    display: block;
}


/* CARDS */

.cards {

    display: grid;

    grid-template-columns:
        repeat(
            4,
            1fr
        );

    gap: 18px;

    margin-bottom: 25px;
}

.card {

    background:
        rgba(255,255,255,.07);

    border:
        1px solid
        rgba(255,255,255,.08);

    border-radius: 20px;

    padding: 23px;

    box-shadow:
        0 15px 40px
        rgba(0,0,0,.2);
}

.card h3 {

    color:
        #aeb5d0;

    font-size: 13px;

    margin: 0 0 12px;
}

.card .value {

    font-size: 30px;

    font-weight: bold;
}


/* GRID */

.grid {

    display: grid;

    grid-template-columns:
        repeat(
            2,
            1fr
        );

    gap: 20px;
}

.panel {

    background:
        rgba(255,255,255,.07);

    border:
        1px solid
        rgba(255,255,255,.08);

    border-radius: 20px;

    padding: 24px;

    margin-bottom: 20px;
}

.panel h2 {

    margin-top: 0;
}


/* BUTTONS */

.primary {

    background:
        linear-gradient(
            135deg,
            #6954ff,
            #9a52e8
        );

    color: white;

    border: none;

    border-radius: 10px;

    padding: 13px 20px;

    cursor: pointer;

    font-weight: bold;
}

.secondary {

    background:
        rgba(255,255,255,.1);

    color: white;

    border: none;

    border-radius: 10px;

    padding: 13px 20px;

    cursor: pointer;
}


/* FORM */

.form-grid {

    display: grid;

    grid-template-columns:
        repeat(
            2,
            1fr
        );

    gap: 15px;
}

.field label {

    display: block;

    color:
        #b7bdd5;

    font-size: 13px;

    margin-bottom: 7px;
}

.field input,
.field select {

    width: 100%;

    padding: 13px;

    border: 1px solid
        rgba(255,255,255,.1);

    border-radius: 10px;

    background:
        rgba(0,0,0,.25);

    color: white;

    outline: none;
}


/* TABLE */

.table-wrap {

    overflow-x: auto;
}

table {

    width: 100%;

    border-collapse:
        collapse;

    font-size: 13px;
}

th,
td {

    padding: 12px;

    border-bottom:
        1px solid
        rgba(255,255,255,.08);

    text-align: left;
}

th {

    color:
        #a8b0ce;

    background:
        rgba(255,255,255,.04);
}


/* RESULT */

.result {

    margin-top: 20px;

    padding: 22px;

    border-radius: 16px;

    background:
        rgba(255,255,255,.07);
}

.result h2 {

    margin-top: 0;
}

.big-result {

    font-size: 32px;

    font-weight: bold;
}

.high {

    color:
        #ff6677;
}

.medium {

    color:
        #ffc857;
}

.low {

    color:
        #53df9a;
}


/* UPLOAD */

.upload-box {

    border:
        2px dashed
        rgba(140,120,255,.5);

    padding: 45px;

    text-align: center;

    border-radius: 20px;

    cursor: pointer;
}

.upload-box input {

    margin-top: 20px;
}


/* MODEL */

.model-grid {

    display: grid;

    grid-template-columns:
        repeat(
            2,
            1fr
        );

    gap: 20px;
}


/* METRIC */

.metric {

    display: flex;

    justify-content:
        space-between;

    padding: 10px 0;

    border-bottom:
        1px solid
        rgba(255,255,255,.08);
}


/* MODULE SELECTION */

.module-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 25px;
    margin-top: 25px;
}

.module-card {
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 22px;
    padding: 30px;
    cursor: pointer;
    transition: .25s;
}

.module-card:hover {
    transform: translateY(-5px);
    background: rgba(118,87,255,.12);
    border-color: rgba(140,120,255,.4);
}

.module-icon {
    font-size: 45px;
    margin-bottom: 15px;
}

.module-card p {
    color: #aeb5d0;
    line-height: 1.6;
    min-height: 80px;
}

/* RISK RECORDS */

.risk-tabs {
    display:flex;
    gap:12px;
    margin:20px 0;
    flex-wrap:wrap;
}

.risk-tab {
    border:none;
    border-radius:12px;
    padding:12px 18px;
    cursor:pointer;
    font-weight:bold;
    color:white;
    background:rgba(255,255,255,.08);
}

.risk-tab.active {
    outline:2px solid rgba(255,255,255,.35);
    transform:translateY(-2px);
}

.risk-tab.high { border-left:4px solid #ff6677; }
.risk-tab.medium { border-left:4px solid #ffc857; }
.risk-tab.low { border-left:4px solid #53df9a; }

.risk-tab span {
    margin-left:6px;
    padding:2px 7px;
    border-radius:10px;
    background:rgba(255,255,255,.12);
}

.risk-scroll {
    max-height:520px;
    overflow-y:auto;
    padding-right:8px;
}

.risk-scroll::-webkit-scrollbar {
    width:8px;
}

.risk-scroll::-webkit-scrollbar-thumb {
    background:rgba(150,130,255,.5);
    border-radius:10px;
}

.risk-record {
    display:grid;
    grid-template-columns:1.1fr 1fr 1fr 1.1fr 1fr;
    gap:15px;
    padding:16px;
    margin-bottom:10px;
    border:1px solid rgba(255,255,255,.08);
    border-radius:14px;
    background:rgba(255,255,255,.05);
    cursor:pointer;
    transition:.2s;
}

.risk-record:hover {
    transform:translateY(-2px);
    background:rgba(118,87,255,.12);
    border-color:rgba(140,120,255,.35);
}

.risk-record small {
    display:block;
    color:#8f96b2;
    margin-bottom:5px;
}

.risk-record strong {
    word-break:break-word;
}

.risk-modal {
    position:fixed;
    inset:0;
    background:rgba(0,0,0,.78);
    display:flex;
    align-items:center;
    justify-content:center;
    z-index:9999;
    padding:20px;
}

.risk-modal-content {
    width:min(920px,96%);
    max-height:90vh;
    overflow-y:auto;
    background:#151827;
    border:1px solid rgba(255,255,255,.12);
    border-radius:20px;
    padding:30px;
    position:relative;
    box-shadow:0 25px 80px rgba(0,0,0,.55);
}

.close-modal {
    position:absolute;
    top:12px;
    right:15px;
    width:40px;
    height:40px;
    border:none;
    border-radius:50%;
    background:rgba(255,255,255,.1);
    color:white;
    font-size:26px;
    cursor:pointer;
}

.detail-grid {
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:14px;
    margin-top:20px;
}

.detail-item {
    padding:14px;
    border-radius:12px;
    background:rgba(255,255,255,.05);
}

.detail-item label {
    display:block;
    color:#9299b6;
    font-size:12px;
    margin-bottom:5px;
}

.factor-box {
    margin-top:20px;
    padding:18px;
    border-radius:14px;
    background:rgba(255,255,255,.06);
}

.chart-grid {
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:20px;
    margin-top:20px;
}

.chart-card {
    background:rgba(255,255,255,.07);
    border:1px solid rgba(255,255,255,.08);
    border-radius:20px;
    padding:20px;
    min-height:330px;
    position:relative;
}

.chart-card canvas {
    width:100% !important;
    height:260px !important;
}

.chart-card h3 {
    margin-top:0;
    color:#dfe3f7;
}

/* Hide charts when there is no data so the dashboard has no blank chart spaces. */
.chart-card.chart-empty {
    display: none !important;
}

.chart-card canvas {
    background: transparent !important;
}

@media(max-width:700px) {
    .risk-record {
        grid-template-columns:1fr 1fr;
    }

    .detail-grid,
    .chart-grid {
        grid-template-columns:1fr;
    }
}

/* RESPONSIVE */

@media(max-width:1000px) {

    .cards {

        grid-template-columns:
            repeat(
                2,
                1fr
            );
    }

    .grid,
    .model-grid {

        grid-template-columns:
            1fr;
    }
}

@media(max-width:700px) {

    .sidebar {

        width: 210px;
    }

    .main {

        margin-left: 210px;

        padding: 18px;
    }

    .cards {

        grid-template-columns:
            1fr;
    }

    .form-grid {

        grid-template-columns:
            1fr;
    }
}

</style>

</head>

<body>


<!-- =====================================================
SIDEBAR
===================================================== -->

<aside class="sidebar">

    <div class="logo">
        🛡️ Fraud<span>Guard</span>
    </div>

    <div id="moduleSidebarTitle"
         style="text-align:center;color:#aeb5d0;font-size:13px;margin-bottom:15px;">
        Select a Module
    </div>

    <div class="nav" id="datasetNav" style="display:none;">

        <button type="button"
                onclick="showPage('upload', this)">
            📁 Dataset Upload
        </button>

        <button type="button"
                onclick="showPage('dashboard', this)">
            📊 Dashboard
        </button>

        <button type="button"
                onclick="showPage('analysis', this)">
            📈 Data Analysis
        </button>

        <button type="button"
                onclick="showPage('models', this)">
            🤖 ML Models
        </button>

        <button type="button"
                onclick="showPage('highrisk', this)">
            📋 Records
        </button>

    </div>

    <div class="nav" id="manualNav" style="display:none;">

        <button type="button"
                onclick="showPage('manual', this)">
            🔍 Manual Prediction
        </button>

        <button type="button"
                onclick="showPage('dashboard', this)">
            📊 Dashboard
        </button>

        <button type="button"
                onclick="showPage('analysis', this)">
            📈 Data Analysis
        </button>

        <button type="button"
                onclick="showPage('models', this)">
            🤖 ML Models
        </button>

        <button type="button"
                onclick="showPage('highrisk', this)">
            📋 Records
        </button>

        <button type="button"
                onclick="showPage('history', this)">
            🧾 Prediction History
        </button>

    </div>

    <button type="button"
            id="backToModules"
            class="secondary"
            style="display:none;width:100%;margin-top:18px;"
            onclick="showModuleChooser()">
        ↩ Change Module
    </button>

    <a class="logout" href="/logout">
        🚪 Logout
    </a>

</aside>


<!-- =====================================================
MAIN
===================================================== -->

<main class="main">

<div class="topbar">

    <div>

        <h1>
            Fraud Transaction Detection
        </h1>

        <p style="color:#aeb5d0;">
            AI-powered financial transaction security
        </p>

    </div>

    <div class="user">
        👤 {{ username }}
    </div>

</div>


<!-- =====================================================
MODULE SELECTION
===================================================== -->

<section id="moduleChooser" class="page active">

    <div class="panel">
        <h2>🛡️ Fraud Transaction Detection Modules</h2>
        <p style="color:#aeb5d0;">
            Select how you want to work with the fraud detection system.
        </p>

        <div class="module-grid">

            <div class="module-card" onclick="chooseModule('dataset')">
                <div class="module-icon">📁</div>
                <h2>Dataset Analysis Module</h2>
                <p>
                    Upload your transaction dataset, analyze the data,
                    view the dashboard, evaluate ML models and inspect
                    high-risk records.
                </p>
                <button type="button" class="primary">
                    Open Dataset Module
                </button>
            </div>

            <div class="module-card" onclick="chooseModule('manual')">
                <div class="module-icon">🔍</div>
                <h2>Manual Prediction Module</h2>
                <p>
                    Enter one transaction manually, get Fraud/Normal
                    prediction, view probability, risk level, dashboard,
                    analysis, ML models, high-risk records and history.
                </p>
                <button type="button" class="primary">
                    Open Manual Module
                </button>
            </div>

        </div>
    </div>

</section>


<!-- =====================================================
DASHBOARD
===================================================== -->

<section
    id="dashboard"
    class="page"
>

<div class="cards">

    <div class="card">

        <h3>
            TOTAL TRANSACTIONS
        </h3>

        <div
            class="value"
            id="totalRecords"
        >
            0
        </div>

    </div>

    <div class="card">

        <h3>
            FRAUD TRANSACTIONS
        </h3>

        <div
            class="value"
            id="fraudRecords"
        >
            0
        </div>

    </div>

    <div class="card">

        <h3>
            NORMAL TRANSACTIONS
        </h3>

        <div
            class="value"
            id="normalRecords"
        >
            0
        </div>

    </div>

    <div class="card">

        <h3>
            HIGH RISK RECORDS
        </h3>

        <div
            class="value"
            id="highRiskRecords"
        >
            0
        </div>

    </div>

</div>


<div class="chart-grid">

    <div class="chart-card">
        <h3>Fraud vs Normal</h3>
        <canvas id="fraudChart"></canvas>
    </div>

    <div class="chart-card">
        <h3>Fraud Rate</h3>
        <canvas id="rateChart"></canvas>
    </div>

    <div class="chart-card">
        <h3>Risk Distribution</h3>
        <canvas id="riskChart"></canvas>
    </div>

    <div class="chart-card">
        <h3>Transactions by Hour</h3>
        <canvas id="hourChart"></canvas>
    </div>

    <div class="chart-card">
        <h3>Fraud by Merchant</h3>
        <canvas id="merchantChart"></canvas>
    </div>

    <div class="chart-card">
        <h3>Fraud by Location</h3>
        <canvas id="locationChart"></canvas>
    </div>

    <div class="chart-card">
        <h3>Transaction Amount Groups</h3>
        <canvas id="amountChart"></canvas>
    </div>

    <div class="chart-card">
        <h3>Transaction Frequency Groups</h3>
        <canvas id="frequencyChart"></canvas>
    </div>

</div>

</section>


<!-- =====================================================
UPLOAD
===================================================== -->

<section
    id="upload"
    class="page active"
>

<div class="panel">

    <h2>
        📁 Upload Transaction Dataset
    </h2>

    <p style="color:#aeb5d0;">
        Upload a CSV or XLSX transaction dataset.
        The system will preprocess the data and train ML models automatically.
    </p>

    <div class="upload-box">

        <div style="font-size:45px;">
            📂
        </div>

        <h3>
            Select Dataset
        </h3>

        <input
            type="file"
            id="datasetFile"
            accept=".csv,.xlsx"
        >

        <br><br>

        <button
            type="button"
            class="primary"
            onclick="uploadDataset()"
        >
            Upload & Train Models
        </button>

    </div>

    <div
        id="uploadMessage"
        style="margin-top:20px;"
    ></div>

</div>

</section>


<!-- =====================================================
ANALYSIS
===================================================== -->

<section
    id="analysis"
    class="page"
>

<div class="panel">

    <h2>
        📈 Dataset Analysis
    </h2>

    <p id="analysisSummary">
        Loading dataset...
    </p>

    <div class="table-wrap">

        <table>

            <thead>

                <tr>

                    <th>Feature</th>
                    <th>Data Type</th>
                    <th>Missing Values</th>
                    <th>Unique Values</th>

                </tr>

            </thead>

            <tbody
                id="analysisTable"
            ></tbody>

        </table>

    </div>

</div>

</section>


<!-- =====================================================
MODELS
===================================================== -->

<section
    id="models"
    class="page"
>

<div
    id="modelsContainer"
    class="model-grid"
></div>

<div class="panel">

    <h2>
        Confusion Matrix
    </h2>

    <div
        id="confusionContainer"
    ></div>

</div>

</section>


<!-- =====================================================
RISK RECORDS
===================================================== -->

<section
    id="highrisk"
    class="page"
>

<div class="panel">

    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:15px;
        flex-wrap:wrap;
    ">

        <div>
            <h2>📋 Records</h2>
            <p style="color:#aeb5d0;">
                Select High Risk, Medium Risk or Low Risk to inspect complete transaction details.
            </p>
        </div>

        <div style="
            display:flex;
            gap:10px;
            flex-wrap:wrap;
            justify-content:flex-end;
        ">

            <a
                id="downloadHighRisk"
                href="/api/risk-download/high?source=dataset"
                class="primary"
                style="text-decoration:none;"
            >
                🔴 Download High Risk
            </a>

            <a
                id="downloadMediumRisk"
                href="/api/risk-download/medium?source=dataset"
                class="primary"
                style="text-decoration:none;"
            >
                🟠 Download Medium Risk
            </a>

            <a
                id="downloadLowRisk"
                href="/api/risk-download/low?source=dataset"
                class="primary"
                style="text-decoration:none;"
            >
                🟢 Download Low Risk
            </a>

        </div>

    </div>

    <div class="risk-tabs">
        <button type="button" class="risk-tab high active"
                onclick="loadRiskRecords('high')">
            🔴 High Risk
            <span id="highCount">0</span>
        </button>

        <button type="button" class="risk-tab medium"
                onclick="loadRiskRecords('medium')">
            🟠 Medium Risk
            <span id="mediumCount">0</span>
        </button>

        <button type="button" class="risk-tab low"
                onclick="loadRiskRecords('low')">
            🟢 Low Risk
            <span id="lowCount">0</span>
        </button>
    </div>

    <div id="riskRecordSummary"
         style="color:#aeb5d0;margin-bottom:15px;">
        Loading risk records...
    </div>

    <div
        id="riskRecordsContainer"
        class="risk-scroll"
    ></div>

</div>

</section>


<!-- =====================================================
RISK DETAILS MODAL
===================================================== -->

<div
    id="riskDetailsModal"
    class="risk-modal"
    style="display:none;"
    onclick="closeRiskDetails(event)"
>

    <div
        class="risk-modal-content"
        onclick="event.stopPropagation()"
    >

        <button
            type="button"
            class="close-modal"
            onclick="closeRiskDetails()"
        >
            ×
        </button>

        <div id="riskDetailsContent"></div>

    </div>

</div>


<!-- =====================================================
MANUAL PREDICTION
===================================================== -->

<section
    id="manual"
    class="page"
>

<div class="panel">

    <h2>
        🔍 Manual Transaction Prediction
    </h2>

    <p style="color:#aeb5d0;">
        Enter transaction information and let the trained ML model estimate fraud risk.
    </p>

    <div class="form-grid">

        <div class="field">

            <label>Account Number</label>

            <input
                id="accountNumber"
                placeholder="ACC1001"
            >

        </div>


        <div class="field">

            <label>Transaction Amount</label>

            <input
                id="amount"
                type="number"
                value="1500"
            >

        </div>


        <div class="field">

            <label>Transaction Hour</label>

            <input
                id="hour"
                type="number"
                min="0"
                max="23"
                value="12"
            >

        </div>


        <div class="field">

            <label>Transaction Frequency</label>

            <input
                id="frequency"
                type="number"
                value="5"
            >

        </div>


        <div class="field">

            <label>Customer Age</label>

            <input
                id="age"
                type="number"
                value="30"
            >

        </div>


        <div class="field">

            <label>Previous Transactions</label>

            <input
                id="previousTransactions"
                type="number"
                value="20"
            >

        </div>


        <div class="field">

            <label>Average Transaction Amount</label>

            <input
                id="averageAmount"
                type="number"
                value="1000"
            >

        </div>


        <div class="field">

            <label>Distance From Usual Location</label>

            <input
                id="distance"
                type="number"
                value="10"
            >

        </div>


        <div class="field">

            <label>Merchant Category</label>

            <select id="merchant">

                <option>Retail</option>
                <option>Food</option>
                <option>Travel</option>
                <option>Online</option>
                <option>Entertainment</option>
                <option>Healthcare</option>
                <option>Electronics</option>

            </select>

        </div>


        <div class="field">

            <label>Location</label>

            <select id="location">

                <option>Coimbatore</option>
                <option>Chennai</option>
                <option>Bangalore</option>
                <option>Mumbai</option>
                <option>Delhi</option>
                <option>Hyderabad</option>
                <option>Madurai</option>

            </select>

        </div>


        <div class="field">

            <label>Transaction Date</label>

            <input
                id="transactionDate"
                type="date"
            >

        </div>


        <div class="field">

            <label>Transaction Time</label>

            <input
                id="transactionTime"
                type="time"
            >

        </div>

    </div>

    <br>

    <button
        type="button"
        class="primary"
        onclick="predictTransaction()"
    >
        🔍 Analyze Transaction
    </button>

    <div
        id="predictionResult"
        class="result"
        style="display:none;"
    ></div>

</div>

</section>


<!-- =====================================================
HISTORY
===================================================== -->

<section
    id="history"
    class="page"
>

<div class="panel">

    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
    ">

        <div>

            <h2>
                🧾 Prediction History
            </h2>

            <p style="color:#aeb5d0;">
                Previous manual transaction predictions.
            </p>

        </div>

        <a
            href="/api/manual/download"
            class="primary"
            style="text-decoration:none;"
        >
            Download Excel
        </a>

    </div>


    <div class="table-wrap">

        <table>

            <thead>

                <tr>

                    <th>Account</th>
                    <th>Amount</th>
                    <th>Prediction</th>
                    <th>Probability</th>
                    <th>Risk</th>
                    <th>Date</th>

                </tr>

            </thead>

            <tbody
                id="historyTable"
            ></tbody>

        </table>

    </div>

</div>

</section>


</main>


<!-- =====================================================
JAVASCRIPT
===================================================== -->

<script>

console.log(
    "FraudGuard AI JavaScript loaded"
);


/* =====================================================
GLOBAL CHART VARIABLES
===================================================== */

let fraudChart = null;
let rateChart = null;
let riskChart = null;
let hourChart = null;
let merchantChart = null;
let locationChart = null;
let amountChart = null;
let frequencyChart = null;


/* =====================================================
SIDEBAR NAVIGATION
===================================================== */

let currentModule = null;

function showModuleChooser() {

    currentModule = null;

    document.querySelectorAll(".page").forEach(
        function(page) {
            page.classList.remove("active");
        }
    );

    document.getElementById(
        "moduleChooser"
    ).classList.add("active");

    document.getElementById(
        "datasetNav"
    ).style.display = "none";

    document.getElementById(
        "manualNav"
    ).style.display = "none";

    document.getElementById(
        "backToModules"
    ).style.display = "none";

    document.getElementById(
        "moduleSidebarTitle"
    ).innerText = "Select a Module";

    document.querySelectorAll(
        ".nav button"
    ).forEach(
        function(button) {
            button.classList.remove("active");
        }
    );
}


function chooseModule(moduleName) {

    currentModule = moduleName;

    document.getElementById(
        "moduleChooser"
    ).classList.remove("active");

    document.getElementById(
        "datasetNav"
    ).style.display =
        moduleName === "dataset"
            ? "flex"
            : "none";

    document.getElementById(
        "manualNav"
    ).style.display =
        moduleName === "manual"
            ? "flex"
            : "none";

    document.getElementById(
        "backToModules"
    ).style.display = "block";

    document.getElementById(
        "moduleSidebarTitle"
    ).innerText =
        moduleName === "dataset"
            ? "Dataset Module"
            : "Manual Prediction Module";

    const firstPage =
        moduleName === "dataset"
            ? "upload"
            : "manual";

    const buttons =
        document.querySelectorAll(
            moduleName === "dataset"
                ? "#datasetNav button"
                : "#manualNav button"
        );

    buttons.forEach(
        function(button) {
            button.classList.remove("active");
        }
    );

    if (buttons.length > 0) {
        buttons[0].classList.add("active");
    }

    showPage(
        firstPage,
        buttons[0]
    );
}


function showPage(
    pageName,
    clickedButton
) {

    console.log(
        "Opening page:",
        pageName,
        "Module:",
        currentModule
    );

    if (!currentModule) {
        return;
    }

    document.querySelectorAll(
        ".page"
    ).forEach(
        function(page) {
            page.classList.remove("active");
        }
    );

    const navSelector =
        currentModule === "dataset"
            ? "#datasetNav button"
            : "#manualNav button";

    document.querySelectorAll(
        navSelector
    ).forEach(
        function(button) {
            button.classList.remove("active");
        }
    );

    const selectedPage =
        document.getElementById(pageName);

    if (!selectedPage) {
        alert(
            "Page not found: " + pageName
        );
        return;
    }

    selectedPage.classList.add("active");

    if (clickedButton) {
        clickedButton.classList.add("active");
    }

    if (pageName === "dashboard") {
        loadDashboard();
    }

    if (pageName === "analysis") {
        loadAnalysis();
    }

    if (pageName === "models") {
        loadModels();
    }

    if (pageName === "highrisk") {
        loadRiskSummary();
        loadHighRisk();
    }

    if (pageName === "history") {
        loadHistory();
    }
}


/* =====================================================
DASHBOARD
===================================================== */

async function loadDashboard() {

    try {

        const endpoint =
            currentModule === "manual"
                ? "/api/manual-dashboard"
                : "/api/dashboard";

        const response =
            await fetch(endpoint);

        const data =
            await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById(
            "totalRecords"
        ).innerText = data.records;

        document.getElementById(
            "fraudRecords"
        ).innerText = data.fraud;

        document.getElementById(
            "normalRecords"
        ).innerText = data.normal;

        document.getElementById(
            "highRiskRecords"
        ).innerText = data.high_risk;

        createFraudChart(
            data.fraud,
            data.normal
        );

        createRateChart(
            data.fraud_rate
        );

        createRiskChart(data.charts?.risk);
        createBarChart("hourChart", data.charts?.hour, "Transactions");
        createBarChart("merchantChart", data.charts?.merchant, "Fraud Transactions");
        createBarChart("locationChart", data.charts?.location, "Fraud Transactions");
        createBarChart("amountChart", data.charts?.amount, "Transactions");
        createBarChart("frequencyChart", data.charts?.frequency, "Transactions");

    } catch (error) {

        console.error(error);

    }
}


function setChartEmptyState(canvas, isEmpty) {
    if (!canvas) return;
    const card = canvas.closest(".chart-card");
    if (card) {
        card.classList.toggle("chart-empty", isEmpty);
        card.style.display = isEmpty ? "none" : "";
    }
    canvas.style.background = "transparent";
}


/* =====================================================
FRAUD CHART
===================================================== */

function createFraudChart(
    fraud,
    normal
) {

    const canvas =
        document.getElementById(
            "fraudChart"
        );

    if (!canvas) return;

    const fraudEmpty = Number(fraud || 0) === 0 && Number(normal || 0) === 0;
    setChartEmptyState(canvas, fraudEmpty);

    if (fraudChart) {

        fraudChart.destroy();

    }

    fraudChart =
        new Chart(
            canvas,
            {

                type: "doughnut",

                data: {

                    labels: [
                        "Fraud",
                        "Normal"
                    ],

                    datasets: [{

                        data: [
                            fraud,
                            normal
                        ]

                    }]

                },

                options: {

                    responsive: true,

                    plugins: {

                        legend: {

                            labels: {
                                color:
                                    "white"
                            }

                        }

                    }

                }

            }
        );

}


/* =====================================================
RATE CHART
===================================================== */

function createRateChart(
    rate
) {

    const canvas =
        document.getElementById(
            "rateChart"
        );

    if (!canvas) return;

    const rateEmpty = rate === null || rate === undefined || Number.isNaN(Number(rate));
    setChartEmptyState(canvas, rateEmpty);

    if (rateChart) {

        rateChart.destroy();

    }

    rateChart =
        new Chart(
            canvas,
            {

                type: "bar",

                data: {

                    labels: [
                        "Fraud Rate"
                    ],

                    datasets: [{

                        label:
                            "Percentage",

                        data: [
                            rate
                        ]

                    }]

                },

                options: {

                    scales: {

                        y: {

                            beginAtZero: true,

                            max: 100,

                            ticks: {
                                color:
                                    "white"
                            }

                        },

                        x: {

                            ticks: {
                                color:
                                    "white"
                            }

                        }

                    },

                    plugins: {

                        legend: {

                            labels: {
                                color:
                                    "white"
                            }

                        }

                    }

                }

            }
        );

}


function destroyChart(chart) {
    if (chart) {
        chart.destroy();
    }
    return null;
}


function createRiskChart(chartData) {

    const canvas = document.getElementById("riskChart");
    if (!canvas) return;

    const riskValues = chartData?.values || [];
    const riskEmpty = !chartData || !riskValues.length || riskValues.every(v => Number(v) === 0);
    setChartEmptyState(canvas, riskEmpty);
    riskChart = destroyChart(riskChart);
    if (riskEmpty) return;

    riskChart = new Chart(canvas, {
        type: "doughnut",
        data: {
            labels: chartData.labels || [],
            datasets: [{
                data: chartData.values || []
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: { color: "white" }
                }
            }
        }
    });
}


function createBarChart(canvasId, chartData, label) {

    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const values = Array.isArray(chartData?.values) ? chartData.values : [];
    const labels = Array.isArray(chartData?.labels) ? chartData.labels : [];

    const chartEmpty =
        labels.length === 0 ||
        values.length === 0 ||
        values.every(v => Number(v) === 0);

    setChartEmptyState(canvas, chartEmpty);

    const chartMap = {
        hourChart: "hourChart",
        merchantChart: "merchantChart",
        locationChart: "locationChart",
        amountChart: "amountChart",
        frequencyChart: "frequencyChart"
    };

    const variableName = chartMap[canvasId];
    if (!variableName) return;

    // The chart variables are declared with let, so window.variableName
    // is NOT the same variable. Destroy the actual chart instance directly.
    const existingChart = {
        hourChart: hourChart,
        merchantChart: merchantChart,
        locationChart: locationChart,
        amountChart: amountChart,
        frequencyChart: frequencyChart
    }[variableName];

    if (existingChart) {
        existingChart.destroy();
    }

    if (chartEmpty) {
        if (variableName === "hourChart") hourChart = null;
        if (variableName === "merchantChart") merchantChart = null;
        if (variableName === "locationChart") locationChart = null;
        if (variableName === "amountChart") amountChart = null;
        if (variableName === "frequencyChart") frequencyChart = null;
        return;
    }

    const newChart = new Chart(canvas, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: values
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: {
                        color: "white",
                        autoSkip: false
                    }
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: "white" }
                }
            },
            plugins: {
                legend: {
                    labels: { color: "white" }
                }
            }
        }
    });

    if (variableName === "hourChart") hourChart = newChart;
    if (variableName === "merchantChart") merchantChart = newChart;
    if (variableName === "locationChart") locationChart = newChart;
    if (variableName === "amountChart") amountChart = newChart;
    if (variableName === "frequencyChart") frequencyChart = newChart;
}


/* =====================================================
UPLOAD DATASET
===================================================== */

async function uploadDataset() {

    const fileInput =
        document.getElementById(
            "datasetFile"
        );

    const message =
        document.getElementById(
            "uploadMessage"
        );

    if (
        !fileInput.files ||
        fileInput.files.length === 0
    ) {

        message.innerHTML =
            "<span style='color:#ff8795;'>Please select a CSV or XLSX file.</span>";

        return;
    }

    const formData =
        new FormData();

    formData.append(
        "file",
        fileInput.files[0]
    );

    message.innerHTML =
        "<span style='color:#ffc857;'>Uploading and training models...</span>";

    try {

        const response =
            await fetch(
                "/api/upload",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            message.innerHTML =
                "<span style='color:#ff8795;'>" +
                data.error +
                "</span>";

            return;
        }

        message.innerHTML =
            "<span style='color:#53df9a;'>" +
            data.message +
            "<br>Records: " +
            data.records +
            "<br>Fraud: " +
            data.fraud +
            "<br>Normal: " +
            data.normal +
            "</span>";


        // Refresh all dataset views immediately after upload/training.
        if (currentModule === "dataset") {
            await loadDashboard();
            await loadAnalysis();
            await loadModels();
            await loadRiskSummary();
            await loadRiskRecords("high");
        }

    } catch (error) {

        message.innerHTML =
            "<span style='color:#ff8795;'>" +
            error.message +
            "</span>";

    }
}


/* =====================================================
DATA ANALYSIS
===================================================== */

async function loadAnalysis() {

    try {

        const endpoint =
            currentModule === "manual"
                ? "/api/manual-analysis"
                : "/api/dataset-analysis";

        const response =
            await fetch(endpoint);

        const data =
            await response.json();

        const table =
            document.getElementById(
                "analysisTable"
            );

        const header =
            document.querySelector(
                "#analysisTable"
            ).closest("table").querySelector("thead tr");

        table.innerHTML = "";

        if (
            currentModule === "manual"
        ) {

            header.innerHTML =
                "<th>Analysis</th>" +
                "<th>Value</th>";

            if (
                !data.records ||
                data.records.length === 0
            ) {
                table.innerHTML =
                    "<tr><td colspan='2'>No manual predictions available.</td></tr>";

                document.getElementById(
                    "analysisSummary"
                ).innerText =
                    "Make a manual prediction first to view manual prediction analysis.";

                return;
            }

            data.records.forEach(
                function(row) {

                    const tr =
                        document.createElement("tr");

                    tr.innerHTML =
                        "<td>" +
                        row.Feature +
                        "</td>" +
                        "<td>" +
                        row.Value +
                        "</td>";

                    table.appendChild(tr);
                }
            );

            document.getElementById(
                "analysisSummary"
            ).innerText =
                "Analysis of transactions entered through the Manual Prediction Module.";

            return;
        }

        header.innerHTML =
            "<th>Analysis</th>" +
            "<th>Value</th>";

        if (data.error) {
            table.innerHTML =
                "<tr><td colspan='2'>" +
                data.error +
                "</td></tr>";
            return;
        }

        const analysisRows = [
            ["Total Records", data.total_records],
            ["Fraud Records", data.fraud_records],
            ["Normal Records", data.normal_records],
            ["Fraud Percentage", data.fraud_percentage + "%"],
            ["Missing Values", data.missing_values],
            ["Duplicate Rows", data.duplicates]
        ];

        Object.entries(
            data.numeric_statistics || {}
        ).forEach(function([column, stats]) {
            analysisRows.push(
                [column + " Mean", stats.mean],
                [column + " Minimum", stats.min],
                [column + " Maximum", stats.max]
            );
        });

        analysisRows.forEach(function(item) {

            const row =
                document.createElement("tr");

            row.innerHTML =
                "<td>" +
                escapeHtml(item[0]) +
                "</td>" +
                "<td>" +
                escapeHtml(item[1]) +
                "</td>";

            table.appendChild(row);
        });

        document.getElementById(
            "analysisSummary"
        ).innerText =
            "Interactive dataset analysis showing transaction counts, fraud rate, data quality and numeric statistics.";

    } catch (error) {

        console.error(error);

    }
}


/* =====================================================
MODELS
===================================================== */

async function loadModels() {

    try {

        const response =
            await fetch(
                "/api/models"
            );

        const data =
            await response.json();

        const container =
            document.getElementById(
                "modelsContainer"
            );

        container.innerHTML = "";

        Object.entries(
            data.models || {}
        ).forEach(
            function([
                name,
                metrics
            ]) {

                const card =
                    document.createElement(
                        "div"
                    );

                card.className =
                    "panel";

                card.innerHTML =

                    "<h2>🤖 " +
                    name +
                    "</h2>" +

                    "<div class='metric'>" +
                    "<span>Accuracy</span>" +
                    "<strong>" +
                    metrics.accuracy +
                    "%</strong>" +
                    "</div>" +

                    "<div class='metric'>" +
                    "<span>Precision</span>" +
                    "<strong>" +
                    metrics.precision +
                    "%</strong>" +
                    "</div>" +

                    "<div class='metric'>" +
                    "<span>Recall</span>" +
                    "<strong>" +
                    metrics.recall +
                    "%</strong>" +
                    "</div>" +

                    "<div class='metric'>" +
                    "<span>F1 Score</span>" +
                    "<strong>" +
                    metrics.f1 +
                    "%</strong>" +
                    "</div>" +

                    "<div class='metric'>" +
                    "<span>ROC-AUC</span>" +
                    "<strong>" +
                    metrics.auc +
                    "%</strong>" +
                    "</div>";

                container.appendChild(
                    card
                );

            }
        );

        const confusionContainer =
            document.getElementById(
                "confusionContainer"
            );

        confusionContainer.innerHTML =
            "";

        Object.entries(
            data.confusion_matrices || {}
        ).forEach(
            function([
                name,
                matrix
            ]) {

                let html =
                    "<h3>" +
                    name +
                    "</h3>" +

                    "<table>" +

                    "<tr>" +
                    "<th></th>" +
                    "<th>Predicted Normal</th>" +
                    "<th>Predicted Fraud</th>" +
                    "</tr>" +

                    "<tr>" +
                    "<th>Actual Normal</th>" +
                    "<td>" +
                    (
                        matrix[0]
                        ? matrix[0][0]
                        : 0
                    ) +
                    "</td>" +

                    "<td>" +
                    (
                        matrix[0]
                        ? matrix[0][1]
                        : 0
                    ) +
                    "</td>" +

                    "</tr>" +

                    "<tr>" +
                    "<th>Actual Fraud</th>" +

                    "<td>" +
                    (
                        matrix[1]
                        ? matrix[1][0]
                        : 0
                    ) +
                    "</td>" +

                    "<td>" +
                    (
                        matrix[1]
                        ? matrix[1][1]
                        : 0
                    ) +
                    "</td>" +

                    "</tr>" +

                    "</table>";

                confusionContainer.innerHTML +=
                    html;

            }
        );

    } catch (error) {

        console.error(
            error
        );

    }

}


async function loadRiskSummary() {

    const source =
        currentModule === "manual"
            ? "manual"
            : "dataset";

    try {

        const response =
            await fetch(
                "/api/risk-summary/" +
                source
            );

        const data =
            await response.json();

        if (data.error) return;

        document.getElementById(
            "highCount"
        ).innerText =
            data["High Risk"] || 0;

        document.getElementById(
            "mediumCount"
        ).innerText =
            data["Medium Risk"] || 0;

        document.getElementById(
            "lowCount"
        ).innerText =
            data["Low Risk"] || 0;

        updateRiskDownloadLinks();

    } catch (error) {
        console.error(error);
    }
}


/* =====================================================
RISK DOWNLOAD LINKS
===================================================== */

function updateRiskDownloadLinks() {

    const source =
        currentModule === "manual"
            ? "manual"
            : "dataset";

    const high = document.getElementById("downloadHighRisk");
    const medium = document.getElementById("downloadMediumRisk");
    const low = document.getElementById("downloadLowRisk");

    if (high) {
        high.href = "/api/risk-download/high?source=" + source;
    }

    if (medium) {
        medium.href = "/api/risk-download/medium?source=" + source;
    }

    if (low) {
        low.href = "/api/risk-download/low?source=" + source;
    }
}


/* =====================================================
RISK RECORDS
===================================================== */

async function loadHighRisk() {
    loadRiskRecords("high");
}


async function loadRiskRecords(level) {

    const source =
        currentModule === "manual"
            ? "manual"
            : "dataset";

    document.querySelectorAll(".risk-tab").forEach(
        button => button.classList.remove("active")
    );

    const activeButton =
        document.querySelector(
            ".risk-tab." + level
        );

    if (activeButton) {
        activeButton.classList.add("active");
    }

    const container =
        document.getElementById(
            "riskRecordsContainer"
        );

    container.innerHTML =
        "<div class='panel'>Loading " +
        level +
        " risk records...</div>";

    try {

        const response =
            await fetch(
                "/api/risk-records/" +
                source +
                "/" +
                level
            );

        const data =
            await response.json();

        if (data.error) {
            container.innerHTML =
                "<div class='panel'>" +
                data.error +
                "</div>";
            return;
        }

        document.getElementById(
            "riskRecordSummary"
        ).innerText =
            data.count +
            " " +
            data.risk +
            " records found. Click any record to view complete details.";

        if (level === "high") {
            document.getElementById("highCount").innerText = data.count;
        }

        if (level === "medium") {
            document.getElementById("mediumCount").innerText = data.count;
        }

        if (level === "low") {
            document.getElementById("lowCount").innerText = data.count;
        }

        if (!data.records || data.records.length === 0) {
            container.innerHTML =
                "<div class='panel'>No " +
                data.risk +
                " records found.</div>";
            return;
        }

        let html = "";

        data.records.forEach(function(record) {

            const riskClass =
                record.risk === "High Risk"
                    ? "high"
                    : record.risk === "Medium Risk"
                        ? "medium"
                        : "low";

            const encoded =
                encodeURIComponent(
                    JSON.stringify(record)
                );

            html += `
                <div class="risk-record"
                     onclick="showRiskDetails(decodeURIComponent('${encoded}'))">

                    <div>
                        <small>Account / ID</small>
                        <strong>${escapeHtml(record.account_number || record.id)}</strong>
                    </div>

                    <div>
                        <small>Amount</small>
                        <strong>₹${Number(record.amount || 0).toLocaleString("en-IN")}</strong>
                    </div>

                    <div>
                        <small>Merchant</small>
                        <strong>${escapeHtml(record.merchant || "N/A")}</strong>
                    </div>

                    <div>
                        <small>Location</small>
                        <strong>${escapeHtml(record.location || "N/A")}</strong>
                    </div>

                    <div>
                        <small>Probability / Risk</small>
                        <strong class="${riskClass}">
                            ${Number(record.probability || 0).toFixed(2)}%
                            · ${escapeHtml(record.risk || "")}
                        </strong>
                    </div>

                </div>
            `;
        });

        container.innerHTML = html;

    } catch (error) {

        console.error(error);

        container.innerHTML =
            "<div class='panel'>Unable to load risk records.</div>";
    }
}


function escapeHtml(value) {

    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


function showRiskDetails(encodedRecord) {

    let record;

    try {
        record =
            typeof encodedRecord === "string"
                ? JSON.parse(encodedRecord)
                : encodedRecord;
    } catch (error) {
        console.error(error);
        return;
    }

    const modal =
        document.getElementById(
            "riskDetailsModal"
        );

    const factors =
        record.factors ||
        "No major suspicious factors detected.";

    const riskClass =
        record.risk === "High Risk"
            ? "high"
            : record.risk === "Medium Risk"
                ? "medium"
                : "low";

    document.getElementById(
        "riskDetailsContent"
    ).innerHTML = `

        <h2>🔎 Complete Transaction Details</h2>

        <p style="color:#aeb5d0;">
            ${record.source === "dataset"
                ? "Dataset Record"
                : "Manual Prediction Record"}
            · ID: ${escapeHtml(record.id)}
        </p>

        <div class="detail-grid">

            <div class="detail-item">
                <label>Account Number / ID</label>
                <strong>${escapeHtml(record.account_number || record.id)}</strong>
            </div>

            <div class="detail-item">
                <label>Transaction Amount</label>
                <strong>₹${Number(record.amount || 0).toLocaleString("en-IN")}</strong>
            </div>

            <div class="detail-item">
                <label>Transaction Date</label>
                <strong>${escapeHtml(record.transaction_date || "N/A")}</strong>
            </div>

            <div class="detail-item">
                <label>Transaction Time</label>
                <strong>${escapeHtml(record.transaction_time || "N/A")}</strong>
            </div>

            <div class="detail-item">
                <label>Transaction Hour</label>
                <strong>${escapeHtml(record.hour)}</strong>
            </div>

            <div class="detail-item">
                <label>Transaction Frequency</label>
                <strong>${escapeHtml(record.frequency)}</strong>
            </div>

            <div class="detail-item">
                <label>Merchant Category</label>
                <strong>${escapeHtml(record.merchant || "N/A")}</strong>
            </div>

            <div class="detail-item">
                <label>Location</label>
                <strong>${escapeHtml(record.location || "N/A")}</strong>
            </div>

            <div class="detail-item">
                <label>Customer Age</label>
                <strong>${escapeHtml(record.age)}</strong>
            </div>

            <div class="detail-item">
                <label>Previous Transactions</label>
                <strong>${escapeHtml(record.previous_transactions)}</strong>
            </div>

            <div class="detail-item">
                <label>Average Amount</label>
                <strong>₹${Number(record.average_amount || 0).toLocaleString("en-IN")}</strong>
            </div>

            <div class="detail-item">
                <label>Distance From Usual Location</label>
                <strong>${escapeHtml(record.distance)}</strong>
            </div>

            <div class="detail-item">
                <label>Prediction</label>
                <strong>${escapeHtml(record.prediction)}</strong>
            </div>

            <div class="detail-item">
                <label>Fraud Probability</label>
                <strong>${Number(record.probability || 0).toFixed(2)}%</strong>
            </div>

            <div class="detail-item">
                <label>Risk Level</label>
                <strong class="${riskClass}">
                    ${escapeHtml(record.risk)}
                </strong>
            </div>

            ${record.actual_fraud !== undefined ? `
            <div class="detail-item">
                <label>Dataset Actual Label</label>
                <strong>${record.actual_fraud === 1 ? "Fraud" : "Normal"}</strong>
            </div>
            ` : ""}

        </div>

        <div class="factor-box">
            <h3>⚠️ Risk Factors</h3>
            <p>${escapeHtml(factors)}</p>
        </div>
    `;

    modal.style.display = "flex";
}


function closeRiskDetails(event) {

    if (
        event &&
        event.target &&
        event.target.id !== "riskDetailsModal"
    ) {
        return;
    }

    document.getElementById(
        "riskDetailsModal"
    ).style.display = "none";
}


/* =====================================================
MANUAL PREDICTION
===================================================== */

async function predictTransaction() {

    const data = {

        account_number:
            document.getElementById(
                "accountNumber"
            ).value,

        amount:
            Number(
                document.getElementById(
                    "amount"
                ).value
            ),

        hour:
            Number(
                document.getElementById(
                    "hour"
                ).value
            ),

        frequency:
            Number(
                document.getElementById(
                    "frequency"
                ).value
            ),

        age:
            Number(
                document.getElementById(
                    "age"
                ).value
            ),

        previous_transactions:
            Number(
                document.getElementById(
                    "previousTransactions"
                ).value
            ),

        average_amount:
            Number(
                document.getElementById(
                    "averageAmount"
                ).value
            ),

        distance:
            Number(
                document.getElementById(
                    "distance"
                ).value
            ),

        merchant:
            document.getElementById(
                "merchant"
            ).value,

        location:
            document.getElementById(
                "location"
            ).value,

        transaction_date:
            document.getElementById(
                "transactionDate"
            ).value,

        transaction_time:
            document.getElementById(
                "transactionTime"
            ).value
    };

    const resultBox =
        document.getElementById(
            "predictionResult"
        );

    resultBox.style.display =
        "block";

    resultBox.innerHTML =
        "Analyzing transaction...";

    try {

        const response =
            await fetch(
                "/api/predict",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            data
                        )
                }
            );

        const result =
            await response.json();

        if (!response.ok) {

            resultBox.innerHTML =
                "<span style='color:#ff8795;'>" +
                result.error +
                "</span>";

            return;
        }

        let riskClass =
            "low";

        if (
            result.risk ===
            "High Risk"
        ) {

            riskClass =
                "high";

        } else if (
            result.risk ===
            "Medium Risk"
        ) {

            riskClass =
                "medium";

        }

        let factors =
            result.factors
            .map(
                factor =>
                    "<li>" +
                    factor +
                    "</li>"
            )
            .join("");

        resultBox.innerHTML =

            "<h2>Transaction Prediction Result</h2>" +

            "<div class='big-result " +
            riskClass +
            "'>" +

            result.prediction +

            "</div>" +

            "<p>" +

            "<strong>Fraud Probability:</strong> " +

            result.probability +

            "%</p>" +

            "<p>" +

            "<strong>Risk Level:</strong> " +

            result.risk +

            "</p>" +

            "<h3>Risk Factors</h3>" +

            "<ul>" +

            factors +

            "</ul>";

    } catch (error) {

        resultBox.innerHTML =
            "<span style='color:#ff8795;'>" +
            error.message +
            "</span>";

    }

}


/* =====================================================
HISTORY
===================================================== */

async function loadHistory() {

    try {

        const response =
            await fetch(
                "/api/history"
            );

        const data =
            await response.json();

        const table =
            document.getElementById(
                "historyTable"
            );

        table.innerHTML = "";

        if (
            !data.records ||
            data.records.length === 0
        ) {

            table.innerHTML =
                "<tr><td colspan='6'>No prediction history yet.</td></tr>";

            return;
        }

        data.records.forEach(
            function(row) {

                let riskClass =
                    "low";

                if (
                    row.risk ===
                    "High Risk"
                ) {

                    riskClass =
                        "high";

                } else if (
                    row.risk ===
                    "Medium Risk"
                ) {

                    riskClass =
                        "medium";

                }

                const tr =
                    document.createElement(
                        "tr"
                    );

                tr.innerHTML =

                    "<td>" +
                    (
                        row.account_number ||
                        "-"
                    ) +
                    "</td>" +

                    "<td>₹" +
                    Number(
                        row.amount || 0
                    ).toFixed(2) +
                    "</td>" +

                    "<td>" +
                    row.prediction +
                    "</td>" +

                    "<td>" +
                    row.probability +
                    "%</td>" +

                    "<td class='" +
                    riskClass +
                    "'>" +
                    row.risk +
                    "</td>" +

                    "<td>" +
                    row.created_at +
                    "</td>";

                table.appendChild(
                    tr
                );

            }
        );

    } catch (error) {

        console.error(
            error
        );

    }

}


/* =====================================================
PAGE LOAD
===================================================== */

document.addEventListener(
    "DOMContentLoaded",
    function() {

        console.log(
            "Application loaded"
        );

        showModuleChooser();

        const dateInput =
            document.getElementById("transactionDate");

        const timeInput =
            document.getElementById("transactionTime");

        if (dateInput && !dateInput.value) {
            dateInput.value =
                new Date().toISOString().slice(0, 10);
        }

        if (timeInput && !timeInput.value) {
            const now = new Date();
            dateInput.value =
                dateInput.value ||
                now.toISOString().slice(0, 10);
            timeInput.value =
                now.toTimeString().slice(0, 5);
        }

    }
);

</script>

</body>
</html>
"""


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "🛡️ FraudGuard AI - Fraud Transaction Detection System"
    )

    print("=" * 60)

    print(
        "Dataset:",
        DATASET
    )

    print(
        "Loading active uploaded dataset if available..."
    )

    load_dataset()

    print(
        "Records:",
        len(dataset_df)
    )

    print(
        "Models trained:",
        list(models.keys())
    )

    print(
        "Open browser:"
    )

    print(
        "http://127.0.0.1:5000" 
    ) 
 
    print("=" * 60) 
 
    app.run( 
        debug=True, 
        host="127.0.0.1", 
        port=5000 
    )
