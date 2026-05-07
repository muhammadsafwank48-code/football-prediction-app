import streamlit as st
import numpy as np
import pickle

# ==========================================
# LOAD MODEL FILES
# ==========================================

model = pickle.load(open("modeldone.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

home_team_encoder = pickle.load(open("home_team_encoder.pkl", "rb"))
away_team_encoder = pickle.load(open("away_team_encoder.pkl", "rb"))
venue_encoder = pickle.load(open("venue_encoder.pkl", "rb"))
referee_encoder = pickle.load(open("referee_encoder.pkl", "rb"))
home_top_scorer_encoder = pickle.load(open("home_top_scorer_encoder.pkl", "rb"))
away_top_scorer_encoder = pickle.load(open("away_top_scorer_encoder.pkl", "rb"))

# ==========================================
# RESULT LABELS
# ==========================================

RESULT_MAP = {
    0: "Home Win",
    1: "Draw",
    2: "Away Win"
}

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Football Match Prediction",
    page_icon="⚽",
    layout="centered"
)

# ==========================================
# TITLE
# ==========================================

st.title("⚽ Football Match Prediction")
st.write("Predict Upcoming Football Match Result")

# ==========================================
# USER INPUTS
# ==========================================

st.subheader("Enter Match Details")

# ---------- NUMERICAL INPUTS ----------

home_form = st.number_input(
    "Home Team League Points",
    min_value=0.0
)

away_form = st.number_input(
    "Away Team League Points",
    min_value=0.0
)

home_shots = st.number_input(
    "Home Team Average Shots",
    min_value=0.0
)

away_shots = st.number_input(
    "Away Team Average Shots",
    min_value=0.0
)

home_possession = st.number_input(
    "Home Team Possession %",
    min_value=0.0,
    max_value=100.0
)

away_possession = st.number_input(
    "Away Team Possession %",
    min_value=0.0,
    max_value=100.0
)

home_yellow = st.number_input(
    "Home Team Average Yellow Cards",
    min_value=0.0
)

away_yellow = st.number_input(
    "Away Team Average Yellow Cards",
    min_value=0.0
)

# ---------- TEXT INPUTS ----------

home_team = st.text_input(
    "Home Team Name"
)

away_team = st.text_input(
    "Away Team Name"
)

venue = st.text_input(
    "Stadium Name"
)

referee = st.text_input(
    "Referee Name"
)

home_top_scorer = st.text_input(
    "Home Team Top Scorer"
)

away_top_scorer = st.text_input(
    "Away Team Top Scorer"
)

# ==========================================
# SAFE ENCODING FUNCTION
# ==========================================

def safe_encode(encoder, value):

    value = value.strip()

    if value in encoder.classes_:
        return encoder.transform([value])[0]

    else:
        return 0

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("Predict Match Result"):

    # ---------- ENCODE TEXT VALUES ----------

    home_team_encoded = safe_encode(
        home_team_encoder,
        home_team
    )

    away_team_encoded = safe_encode(
        away_team_encoder,
        away_team
    )

    venue_encoded = safe_encode(
        venue_encoder,
        venue
    )

    referee_encoded = safe_encode(
        referee_encoder,
        referee
    )

    home_top_scorer_encoded = safe_encode(
        home_top_scorer_encoder,
        home_top_scorer
    )

    away_top_scorer_encoded = safe_encode(
        away_top_scorer_encoder,
        away_top_scorer
    )

    # ==========================================
    # INPUT ARRAY
    # ==========================================

    input_data = [
        home_form,
        away_form,
        home_shots,
        away_shots,
        home_possession,
        away_possession,
        home_yellow,
        away_yellow,
        home_team_encoded,
        away_team_encoded,
        venue_encoded,
        referee_encoded,
        home_top_scorer_encoded,
        away_top_scorer_encoded
    ]

    # ==========================================
    # CONVERT TO NUMPY
    # ==========================================

    X = np.array(input_data).reshape(1, -1)

    # ==========================================
    # SCALE ONLY NUMERICAL COLUMNS
    # ==========================================

    numerical_count = 8

    X[:, :numerical_count] = scaler.transform(
        X[:, :numerical_count]
    )

    # ==========================================
    # PREDICTION
    # ==========================================

    prediction = int(model.predict(X)[0])

    probabilities = model.predict_proba(X)[0]

    confidence = round(
        max(probabilities) * 100,
        2
    )

    # ==========================================
    # OUTPUT
    # ==========================================

    st.success(
        f"Prediction: {RESULT_MAP[prediction]}"
    )

    st.info(
        f"Confidence: {confidence}%"
    )

    # ==========================================
    # PROBABILITY SECTION
    # ==========================================

    st.subheader("Winning Probabilities")

    for i, prob in enumerate(probabilities):

        st.write(
            f"{RESULT_MAP[i]} : {round(prob * 100, 2)}%"
        )