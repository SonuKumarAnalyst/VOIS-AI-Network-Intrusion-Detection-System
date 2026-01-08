import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from groq import Groq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    layout="wide"
)

st.title("🛡️ AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Cybersecurity Project**  
Machine Learning (**Random Forest**) + **Generative AI (Groq)**  
Detects network attacks and explains them in simple language.
""")

# ---------------- CONFIG ----------------
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")
groq_api_key = st.sidebar.text_input(
    "Groq API Key (starts with gsk_)",
    type="password"
)
st.sidebar.caption("Optional – used for AI explanation")

st.sidebar.header("🤖 Model Training")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, nrows=15000)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# ---------------- TRAIN MODEL ----------------
def train_model(df):
    features = [
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Total Length of Fwd Packets',
        'Fwd Packet Length Max',
        'Flow IAT Mean',
        'Flow IAT Std',
        'Flow Packets/s'
    ]
    target = 'Label'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, features, X_test, y_test

# ---------------- LOAD DATASET ----------------
try:
    df = load_data(DATA_FILE)
    st.sidebar.success(f"Dataset Loaded: {len(df)} rows")
except:
    st.error("❌ Dataset file not found. Upload CSV to project folder.")
    st.stop()

# ---------------- DATASET OVERVIEW ----------------
st.header("📊 Dataset Overview")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Attack vs Benign Distribution")
    st.bar_chart(df['Label'].value_counts())

with c2:
    st.subheader("Statistical Summary")
    st.dataframe(
        df.describe().loc[['mean', 'std', 'min', 'max']],
        use_container_width=True
    )

# ---------------- TRAIN BUTTON ----------------
if st.sidebar.button("🚀 Train Model Now"):
    with st.spinner("Training model..."):
        model, acc, features, X_test, y_test = train_model(df)

        st.session_state.model = model
        st.session_state.features = features
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.feature_importance = model.feature_importances_

        st.sidebar.success(f"Model Trained | Accuracy: {acc:.2%}")

# ---------------- DASHBOARD ----------------
st.header("📡 Threat Analysis Dashboard")

if "model" in st.session_state:

    # -------- Feature Importance --------
    st.subheader("🔍 Feature Importance")
    fi_df = pd.DataFrame({
        "Feature": st.session_state.features,
        "Importance": st.session_state.feature_importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(fi_df.set_index("Feature"))

    col1, col2 = st.columns(2)

    # -------- Packet Simulation --------
    with col1:
        st.subheader("🎲 Packet Simulation")
        if st.button("Capture Random Packet"):
            idx = np.random.randint(0, len(st.session_state.X_test))
            st.session_state.packet = st.session_state.X_test.iloc[idx]
            st.session_state.actual = st.session_state.y_test.iloc[idx]

    # -------- Analysis --------
    if "packet" in st.session_state:
        packet = st.session_state.packet

        with col1:
            st.write("📄 Packet Features")
            st.dataframe(packet, use_container_width=True)

        with col2:
            st.subheader("🚨 Detection Result")

            prediction = st.session_state.model.predict([packet])[0]
            proba = st.session_state.model.predict_proba([packet])[0]
            classes = st.session_state.model.classes_

            if prediction == "BENIGN":
                st.success("STATUS: SAFE (BENIGN)")
            else:
                st.error(f"ATTACK DETECTED: {prediction}")

            st.caption(f"Ground Truth: {st.session_state.actual}")

            # -------- Confidence Graph --------
            conf_df = pd.DataFrame({
                "Confidence": proba
            }, index=classes)

            st.subheader("📈 Prediction Confidence")
            st.bar_chart(conf_df)

        # -------- Packet vs Dataset Comparison --------
        st.subheader("📡 Packet vs Dataset Mean Comparison")

        packet_df = packet.to_frame("Selected Packet")
        mean_df = df[st.session_state.features].mean().to_frame("Dataset Mean")

        compare_df = pd.concat([packet_df, mean_df], axis=1)
        st.line_chart(compare_df)

        # -------- Groq AI Explanation --------
        st.subheader("🧠 Ask AI Analyst (Groq)")

        if st.button("Generate AI Explanation"):
            if not groq_api_key:
                st.warning("Please enter Groq API Key in sidebar")
            else:
                with st.spinner("Groq is analyzing..."):
                    client = Groq(api_key=groq_api_key)

                    prompt = f"""
                    You are a cybersecurity analyst.
                    The packet was detected as {prediction}.

                    Packet details:
                    {packet.to_string()}

                    Explain simply for a student:
                    - Why this packet looks like {prediction}
                    - Mention key features involved
                    """

                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.6
                    )

                    st.info(response.choices[0].message.content)

else:
    st.info("⬅️ Train the model from sidebar to begin analysis.")
