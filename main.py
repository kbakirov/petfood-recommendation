import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
from collections import Counter

# -----------------------------------
# 1) STREAMLIT PAGE CONFIG
# -----------------------------------

st.set_page_config(page_title="Dog Diet Recommendation", layout="centered")

# -----------------------------------
# 2) CACHED DATA LOADING
# -----------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    food = pd.read_csv("FINAL_COMBINED.csv")
    disease = pd.read_csv("Disease.csv")
    return food, disease

food_df, disease_df = load_data()

# -----------------------------------
# 3) PREPROCESSING FUNCTIONS
# -----------------------------------

def classify_breed_size(row):
    w = (row["min_weight"] + row["max_weight"]) / 2
    if w <= 10:
        return "Small Breed"
    elif w <= 25:
        return "Medium Breed"
    else:
        return "Large Breed"

@st.cache_data(show_spinner=False)
def preprocess_disease(df):
    df = df.copy()
    df["breed_size_category"] = df.apply(classify_breed_size, axis=1)
    return df

disease_df = preprocess_disease(disease_df)

@st.cache_data(show_spinner=False)
def preprocess_food(df):
    df = df.copy()
    nutrients = [
        "protein", "fat", "carbohydrate (nfe)", "crude fibre", "calcium",
        "phospohorus", "potassium", "sodium", "magnesium", "vitamin e",
        "vitamin c", "omega-3-fatty acids", "omega-6-fatty acids",
    ]
    for col in nutrients:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "")
            .str.replace("IU/kg", "")
            .str.extract(r"([\d\.]+)")
            .astype(float)
            .fillna(0.0)
        )

    df["combined_text"] = (
        df["ingredients"].fillna("")
        .str.cat(df["key benefits"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["product title"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["product description"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["helpful tips"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["need/preference"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["alternate product recommendation"].fillna(""), sep=" ", na_rep="")
    )
    return df

food_df = preprocess_food(food_df)

# -----------------------------------
# 4) TEXT VECTORIZATION & SVD
# -----------------------------------

@st.cache_resource(show_spinner=False)
def build_text_pipeline(corpus, n_components=100):
    vect = TfidfVectorizer(stop_words="english", max_features=5000)
    X_tfidf = vect.fit_transform(corpus)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)

    return vect, svd, X_reduced

vectorizer, svd, X_text_reduced = build_text_pipeline(food_df["combined_text"], n_components=100)

# -----------------------------------
# 5) CATEGORICAL ENCODING
# -----------------------------------

@st.cache_resource(show_spinner=False)
def build_categorical_encoder(df):
    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    cats = df[["breed size", "lifestage"]].fillna("Unknown")
    enc.fit(cats)
    return enc, enc.transform(cats)

encoder, X_categorical = build_categorical_encoder(food_df)

# -----------------------------------
# 6) COMBINE FEATURES INTO SPARSE MATRIX
# -----------------------------------

@st.cache_resource(show_spinner=False)
def combine_features(text_reduced, _cat_matrix):
    # Turn dense text_reduced into sparse form
    X_sparse_text = csr_matrix(text_reduced)
    return hstack([X_sparse_text, _cat_matrix])

X_combined = combine_features(X_text_reduced, X_categorical)

# -----------------------------------
# 7) TRAIN RIDGE REGRESSORS FOR NUTRIENTS
# -----------------------------------

@st.cache_resource(show_spinner=False)
def train_nutrient_models(food, _X):
    nutrient_models = {}
    scalers = {}

    nutrients = [
        "protein", "fat", "carbohydrate (nfe)", "crude fibre", "calcium",
        "phospohorus", "potassium", "sodium", "magnesium", "vitamin e",
        "vitamin c", "omega-3-fatty acids", "omega-6-fatty acids",
    ]
    to_scale = {
        "sodium",
        "omega-3-fatty acids",
        "omega-6-fatty acids",
        "calcium",
        "phospohorus",
        "potassium",
        "magnesium",
    }

    for nutrient in nutrients:
        y = food[nutrient].fillna(food[nutrient].median()).values.reshape(-1, 1)
        if nutrient in to_scale:
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y).ravel()
        else:
            scaler = None
            y_scaled = y.ravel()

        X_train, _, y_train, _ = train_test_split(_X, y_scaled, test_size=0.2, random_state=42)

        base = Ridge()
        search = GridSearchCV(
            base,
            param_grid={"alpha": [0.1, 1.0]},
            scoring="r2",
            cv=2,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        nutrient_models[nutrient] = search.best_estimator_
        scalers[nutrient] = scaler

    return nutrient_models, scalers

# **This line must run at import-time** so ridge_models is defined before you use it below:
ridge_models, scalers = train_nutrient_models(food_df, X_combined)

# -----------------------------------
# 8) TRAIN RIDGE CLASSIFIERS FOR INGREDIENT PRESENCE
# -----------------------------------

@st.cache_resource(show_spinner=False)
def train_ingredient_models(food, _X):
    all_ings = []
    for txt in food["ingredients"].dropna():
        tokens = [i.strip().lower() for i in txt.split(",")]
        all_ings.extend(tokens)

    counts = Counter(all_ings)
    frequent = [ing for ing, cnt in counts.items() if cnt >= 5]

    targets = {}
    low = food["ingredients"].fillna("").str.lower()
    for ing in frequent:
        targets[ing] = low.apply(lambda s: int(ing in s)).values

    ing_models = {}
    for ing, y in targets.items():
        clf = RidgeClassifier()
        clf.fit(_X, y)
        ing_models[ing] = clf

    return ing_models, frequent

# **This line must run at import-time** so ingredient_models is defined before you use it below:
ingredient_models, frequent_ingredients = train_ingredient_models(food_df, X_combined)

# -----------------------------------
# 9) DISORDER KEYWORDS DICTIONARY
# -----------------------------------

disorder_keywords = {
    "Inherited musculoskeletal disorders": "joint mobility glucosamine arthritis cartilage flexibility",
    "Inherited gastrointestinal disorders": "digest stomach bowel sensitive diarrhea gut ibs",
    "Inherited endocrine disorders": "thyroid metabolism weight diabetes insulin hormone glucose",
    "Inherited eye disorders": "vision eye retina cataract antioxidant sight ocular",
    "Inherited nervous system disorders": "brain seizure cognitive nerve neuro neurological cognition",
    "Inherited cardiovascular disorders": "heart cardiac circulation omega-3 blood pressure vascular",
    "Inherited skin disorders": "skin allergy itch coat omega-6 dermatitis eczema flaky",
    "Inherited immune disorders": "immune defense resistance inflammatory autoimmune",
    "Inherited urinary and reproductive disorders": "urinary bladder kidney renal urine reproductive",
    "Inherited respiratory disorders": "breath respiratory airway lung cough breathing nasal",
    "Inherited blood disorders": "anemia blood iron hemoglobin platelets clotting hemophilia",
}

# -----------------------------------
# 10) STREAMLIT UI LAYOUT
# -----------------------------------

st.sidebar.title("🐶 Smart Dog Diet Advisor")
st.sidebar.write("Select breed + disorder → get personalized food suggestions")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=80)

st.header("Dog Diet Recommendation")

breed_list = sorted(disease_df["Breed"].unique())
user_breed = st.selectbox("Select dog breed:", breed_list)

if user_breed:
    info = disease_df[disease_df["Breed"] == user_breed]
    if not info.empty:
        breed_size = info["breed_size_category"].values[0]
        disorders = info["Disease"].unique().tolist()
        selected_disorder = st.selectbox("Select disorder:", disorders)
        disorder_type = info[info["Disease"] == selected_disorder]["Disorder"].values[0]

        if st.button("Generate Recommendation"):
            # 10.1) Build query vector
            keywords = disorder_keywords.get(disorder_type, selected_disorder).lower()
            kw_tfidf = vectorizer.transform([keywords])
            kw_reduced = svd.transform(kw_tfidf)

            # One-hot for (breed_size, "Adult")
            cat_vec = encoder.transform([[breed_size, "Adult"]])
            kw_combined = hstack([csr_matrix(kw_reduced), cat_vec])

            # 10.2) Predict nutrients
            nutrient_preds = {}
            for nut, model in ridge_models.items():
                pred = model.predict(kw_combined)[0]
                sc = scalers.get(nut)
                if sc:
                    pred = sc.inverse_transform([[pred]])[0][0]
                nutrient_preds[nut] = round(pred, 2)

            # 10.3) Rank ingredients
            ing_scores = {
                ing: clf.decision_function(kw_combined)[0]
                for ing, clf in ingredient_models.items()
            }
            top_ings = sorted(ing_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            top_ings = [i.title() for i, _ in top_ings]

            # 10.4) Top-3 similar products
            mask = (
                food_df["breed size"].str.lower() == breed_size.lower()
            ) | (food_df["breed size"].str.lower() == "unknown")
            candidates = food_df[mask]
            sims = cosine_similarity(
                kw_tfidf, vectorizer.transform(candidates["combined_text"])
            ).flatten()
            top3_idx = sims.argsort()[-3:][::-1]
            top3_products = candidates.iloc[top3_idx]["product title"].dropna().tolist()

            # 10.5) Display
            st.subheader("🌿 Recommended Ingredients")
            st.write(f"Based on disorder: **{disorder_type}**")
            for ing in top_ings:
                st.write("• " + ing)

            st.subheader("📦 Top 3 Product Matches")
            for prod in top3_products:
                st.write("• " + prod)

            st.subheader("🧪 Nutrient Forecast (% of dry matter)")
            cols = st.columns(3)
            i = 0
            for nut, val in nutrient_preds.items():
                with cols[i % 3]:
                    st.metric(label=nut.title(), value=f"{val} %")
                i += 1

    else:
        st.info("No disease info found for this breed.")
else:
    st.info("Please select a breed to continue.")
