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

st.set_page_config(page_title="Dog Diet Recommendation", layout="wide")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=100)
    st.markdown("### Smart Dog Diet Advisor")
    st.write("Enter breed, choose disorder, and receive personalized food suggestions.")

st.markdown("""
    <style>
        .stApp { background-color: #333333; }
        .block-container { padding-top: 5rem; padding-bottom: 2rem; }
        .stDataFrame, .stTable { background-color: white; border-radius: 10px; }
        .stButton>button { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 8px; height: 3em; width: 100%; }
        .stSelectbox label, .stTextInput label { font-weight: 600; font-size: 1rem; }
    </style>
""", unsafe_allow_html=True)

food_df = pd.read_csv("FINAL_COMBINED.csv")
disease_df = pd.read_csv("Disease.csv")

def classify_breed_size(row):
    weight = (row['min_weight'] + row['max_weight']) / 2
    if weight <= 10:
        return 'Small Breed'
    elif 10 < weight <= 25:
        return 'Medium Breed'
    else:
        return 'Large Breed'

disease_df['breed_size_category'] = disease_df.apply(classify_breed_size, axis=1)

selected_nutrients = [
    'protein', 'fat', 'carbohydrate (nfe)', 'crude fibre', 'calcium',
    'phospohorus', 'potassium', 'sodium', 'magnesium', 'vitamin e',
    'vitamin c', 'omega-3-fatty acids', 'omega-6-fatty acids'
]

for col in selected_nutrients:
    food_df[col] = food_df[col].astype(str).str.replace('%', '').str.replace('IU/kg', '').str.extract(r'([\d.]+)').astype(float)

food_df['combined_text'] = (
    food_df['ingredients'].fillna('') * 3 + ' ' +
    food_df['key benefits'].fillna('') * 2 + ' ' +
    food_df['product title'].fillna('') + ' ' +
    food_df['product description'].fillna('') + ' ' +
    food_df['helpful tips'].fillna('') + ' ' +
    food_df['need/preference'].fillna('') + ' ' +
    food_df['alternate product recommendation'].fillna('')
)

vectorizer = TfidfVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(food_df['combined_text'])
svd = TruncatedSVD(n_components=300, random_state=42)
X_text_reduced = svd.fit_transform(X_text)

encoder = OneHotEncoder(sparse_output=True)
X_categorical = encoder.fit_transform(food_df[['breed size', 'lifestage']].fillna('Unknown'))
X_combined = hstack([csr_matrix(X_text_reduced), X_categorical])

scale_nutrients = ['sodium', 'omega-3-fatty acids', 'omega-6-fatty acids', 'calcium', 'phospohorus', 'potassium', 'magnesium']
scalers = {}
ridge_models = {}

for nutrient in selected_nutrients:
    y = food_df[nutrient].fillna(food_df[nutrient].median())
    if nutrient in scale_nutrients:
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        scalers[nutrient] = scaler_y
    else:
        y_scaled = y
        scalers[nutrient] = None

    X_train, _, y_train, _ = train_test_split(X_combined, y_scaled, test_size=0.2, random_state=42)
    model = Ridge()
    grid = GridSearchCV(model, {'alpha': [0.01, 0.1, 1.0]}, scoring='r2', cv=3)
    grid.fit(X_train, y_train)
    ridge_models[nutrient] = grid.best_estimator_

all_ingredients_flat = []
for ing_list in food_df['ingredients'].dropna():
    ings = [i.strip().lower() for i in ing_list.split(',')]
    all_ingredients_flat.extend(ings)

ingredient_counter = Counter(all_ingredients_flat)
frequent_ingredients = [ingredient for ingredient, count in ingredient_counter.items() if count >= 5]

ingredient_targets = {ing: food_df['ingredients'].fillna('').apply(lambda x: int(ing in x.lower())) for ing in frequent_ingredients}

ingredient_models = {}
for ing in frequent_ingredients:
    y = ingredient_targets[ing]
    model = RidgeClassifier()
    model.fit(X_combined, y)
    ingredient_models[ing] = model

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
    "Inherited blood disorders": "anemia blood iron hemoglobin platelets clotting hemophilia"
}

breed_names = sorted(disease_df['Breed'].unique())
user_breed = st.selectbox("Select dog breed:", breed_names)
breed_info = disease_df[disease_df['Breed'] == user_breed]

if not breed_info.empty:
    breed_size = breed_info['breed_size_category'].values[0]
    disorder_options = breed_info['Disease'].unique()
    selected_disorder = st.selectbox("Select disorder:", disorder_options)
    disorder_type = breed_info[breed_info['Disease'] == selected_disorder]['Disorder'].values[0]

    if st.button("🎯 Generate Personalized Recommendation"):
        keyword_string = disorder_keywords.get(disorder_type, selected_disorder)
        keyword_vec = vectorizer.transform([keyword_string])
        keyword_reduced = svd.transform(keyword_vec)
        keyword_combined = hstack([csr_matrix(keyword_reduced), encoder.transform([[breed_size, 'Adult']])])

        nutrient_forecast = {}
        for nutrient, model in ridge_models.items():
            pred = model.predict(keyword_combined)[0]
            if nutrient in scalers:
                pred = scalers[nutrient].inverse_transform([[pred]])[0][0] if scalers[nutrient] is not None else pred
            nutrient_forecast[nutrient] = round(pred, 2)

        ingredient_scores = {
            ing: model.decision_function(keyword_combined)[0]
            for ing, model in ingredient_models.items()
        }
        top_ingredients = sorted(ingredient_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        final_ingredients = [ing.title() for ing, _ in top_ingredients]

        filtered_products = food_df[
            (food_df['breed size'].str.lower() == breed_size.lower()) |
            (food_df['breed size'].str.lower() == 'unknown')
        ]
        similarities = cosine_similarity(keyword_vec, vectorizer.transform(filtered_products['combined_text'])).flatten()
        top_indices = similarities.argsort()[-3:][::-1]
        recommended_products = filtered_products.iloc[top_indices]['product title'].dropna().tolist()

        st.markdown("## 🧾 Personalized Diet Plan")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
                <div style="background-color:#9abee6;padding:20px;border-radius:10px;text-align:center">
                    <h3 style="color:#2e7d32;">🌿 Recommended Ingredients</h3>
                    <p style="color:#1b5e20;font-size:14px;">Based on disorder: <b>{}</b></p>
                    <ul style="text-align:left;font-size:15px;">
                        {}
                    </ul>
                </div>
            """.format(
                disorder_type,
                "".join([f"<li>{i}</li>" for i in final_ingredients])
            ), unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div style="background-color:#9abee6;padding:20px;border-radius:10px;text-align:center">
                    <h3 style="color:#ef6c00;">📦 Product Suggestions</h3>
                    <p style="color:#e65100;font-size:14px;">Top 3 Matches</p>
                    <ul style="text-align:left;font-size:15px;">
                        {}
                    </ul>
                </div>
            """.format(
                "".join([f"<li>{p}</li>" for p in recommended_products])
            ), unsafe_allow_html=True)

        st.markdown("### 🧪 Nutrient Forecast (% of dry matter)")

        nutrient_list = list(nutrient_forecast.items())
        chunks = [nutrient_list[i:i + 3] for i in range(0, len(nutrient_list), 3)]
        for chunk in chunks:
            cols = st.columns(len(chunk))
            for i, (nutrient, value) in enumerate(chunk):
                with cols[i]:
                    st.metric(label=nutrient.title(), value=f"{value} %")
                    bar_color = "#81c784" if value < 25 else "#ffb74d" if value < 60 else "#e57373"
                    st.markdown(f"""
                        <div style="height: 15px; background: #eee; border-radius: 7px;">
                            <div style="width: {min(value, 100)}%; background: {bar_color}; height: 100%; border-radius: 7px;"></div>
                        </div>
                    """, unsafe_allow_html=True)
