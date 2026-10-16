
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AC Quality Intelligence",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# 2. Load reduced model artifacts
# --------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(PROJECT_DIR / "ac_defect_model_reduced.pkl")
    preprocessor = joblib.load(PROJECT_DIR / "ac_preprocessor_reduced.pkl")
    threshold = float(
        joblib.load(PROJECT_DIR / "ac_threshold_reduced.pkl")
    )
    features = joblib.load(PROJECT_DIR / "ac_features_reduced.pkl")

    return model, preprocessor, threshold, features


# --------------------------------------------------
# 3. Custom styling
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 5% 0%, #e6f3ff 0%, transparent 28%),
        linear-gradient(135deg, #f5f8ff 0%, #eef3fa 100%);
    color: #14213d;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.hero {
    background: linear-gradient(120deg, #102447 0%, #164e8c 65%, #087eaa 100%);
    padding: 32px 36px;
    border-radius: 22px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 12px 35px rgba(19, 55, 100, 0.16);
}

.hero h1 {
    color: white;
    font-size: 31px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

.hero p {
    color: #d8eaff;
    font-size: 14px;
    line-height: 1.7;
    margin: 9px 0 0;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.23);
    color: #e9f7ff;
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 13px;
}

.section-title {
    font-size: 19px;
    font-weight: 800;
    color: #14213d;
    margin: 4px 0;
}

.section-subtitle {
    color: #71809a;
    font-size: 12px;
    line-height: 1.7;
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(160deg, #ffffff 0%, #f5f9ff 100%);
    border: 1px solid #dce7f6;
    border-radius: 22px;
    padding: 27px;
    box-shadow: 0 10px 30px rgba(30, 66, 115, 0.08);
}

.result-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.4px;
    color: #8290a6;
}

.result-name {
    font-size: 27px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 8px;
    letter-spacing: -.8px;
    overflow-wrap: anywhere;
}

.result-desc {
    color: #73819a;
    font-size: 12px;
    line-height: 1.8;
}

.status-pill {
    display: inline-block;
    border-radius: 30px;
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .4px;
}

.good-pill {
    color: #08794e;
    background: #dff8eb;
}

.bad-pill {
    color: #b42335;
    background: #ffe5e7;
}

.divider {
    height: 1px;
    background: #e6edf6;
    margin: 21px 0;
}

.info-row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin: 13px 0;
    font-size: 12px;
}

.info-row .label {
    color: #78869d;
}

.info-row .value {
    color: #192c4c;
    font-weight: 700;
    text-align: right;
}

.footer {
    text-align: center;
    color: #8996aa;
    font-size: 11px;
    line-height: 1.8;
    padding-top: 28px;
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #e1e9f5;
    border-radius: 19px;
    padding: 24px;
    box-shadow: 0 7px 24px rgba(28, 53, 90, 0.055);
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #1765d1, #087eaa);
    color: white;
    border: none;
    border-radius: 11px;
    min-height: 48px;
    font-weight: 800;
    transition: all .2s ease;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(90deg, #1256b7, #076b91);
    color: white;
    box-shadow: 0 7px 17px rgba(23, 101, 209, .22);
}

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label {
    color: #3e4f6b;
    font-size: 12px;
    font-weight: 600;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    border-radius: 9px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. Header
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-tag">❄ &nbsp; QUALITY CONTROL</div>
    <h1>AC Quality Intelligence</h1>
    <p>
        Smart manufacturing quality screening.
        Enter the unit measurements to review the predicted result.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 5. Load artifacts safely
# --------------------------------------------------
try:
    model, preprocessor, threshold, features = load_model_artifacts()
except Exception as exc:
    st.error(
        "Unable to load the prediction files. Please check that all four "
        "reduced-model files are present in the project root."
    )
    st.stop()


# --------------------------------------------------
# 6. Main layout
# --------------------------------------------------
left, right = st.columns([1.15, 0.85], gap="large")


# --------------------------------------------------
# 7. Quality input form
# --------------------------------------------------
with left:
    st.markdown(
        '<div class="section-title">Unit Quality Parameters</div>'
        '<div class="section-subtitle">'
        'Enter the measured values from the AC inspection process.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("ac_quality_form"):
        c1, c2 = st.columns(2)

        with c1:
            pcb_test = st.selectbox(
                "PCB Test Result",
                ["Pass", "Fail"]
            )

            insulation = st.number_input(
                "Insulation Resistance (MΩ)",
                min_value=0.0,
                value=100.0,
                step=1.0
            )

            operator_experience = st.number_input(
                "Operator Experience (years)",
                min_value=0.0,
                value=5.0,
                step=1.0
            )

            vacuum_pressure = st.number_input(
                "Vacuum Pressure (microns)",
                min_value=0.0,
                value=200.0,
                step=10.0
            )

        with c2:
            vibration = st.number_input(
                "Vibration Level (mm/s)",
                min_value=0.0,
                value=2.5,
                step=0.1
            )

            supplier = st.text_input(
                "Component Supplier",
                value="Supplier_A",
                help="Enter a supplier category present in your training data."
            )

            shift = st.selectbox(
                "Production Shift",
                ["Day", "Night"]
            )

            machine_age = st.number_input(
                "Machine Age (years)",
                min_value=0.0,
                value=3.0,
                step=1.0
            )

        submitted = st.form_submit_button(
            "✦   Predict Defect",
            use_container_width=True
        )


# --------------------------------------------------
# 8. Perform prediction
# --------------------------------------------------
if "ac_prediction_result" not in st.session_state:
    st.session_state.ac_prediction_result = None

if submitted:
    if not supplier.strip():
        st.session_state.ac_prediction_result = None
        st.error("Please enter a component supplier.")
    else:
        payload = {
            "PCB_Test_Result": pcb_test,
            "Insulation_Resistance_MOhm": insulation,
            "Operator_Experience_Years": operator_experience,
            "Vacuum_Pressure_Microns": vacuum_pressure,
            "Vibration_Level_mm_s": vibration,
            "Component_Supplier": supplier.strip(),
            "Shift": shift,
            "Machine_Age_Years": machine_age
        }

        try:
            with st.spinner("Analyzing quality measurements..."):
                input_df = pd.DataFrame(
                    [[payload[feature] for feature in features]],
                    columns=features
                )

                input_processed = preprocessor.transform(input_df)

                probability = float(
                    model.predict_proba(input_processed)[0][1]
                )

                prediction = (
                    "Defective"
                    if probability >= threshold
                    else "Not Defective"
                )

                st.session_state.ac_prediction_result = {
                    "defect_probability": probability,
                    "prediction": prediction
                }

        except Exception as exc:
            st.session_state.ac_prediction_result = None
            st.error(
                "Prediction failed. Check that your input categories match "
                "the training data and that the saved artifacts are compatible."
            )
            st.exception(exc)


# --------------------------------------------------
# 9. Prediction dashboard
# --------------------------------------------------
with right:
    st.markdown("""
    <div class="section-title">Prediction Dashboard</div>
    <div class="section-subtitle">
        Review the result of your quality assessment.
    </div>
    """, unsafe_allow_html=True)

    result = st.session_state.ac_prediction_result

    if result:
        probability = float(result["defect_probability"])
        prediction = result["prediction"]
        is_defective = prediction == "Defective"

        if is_defective:
            status_color = "#b42335"
            status_icon = "⚠"
            pill_class = "bad-pill"
            pill_text = "REVIEW REQUIRED"
            status_message = (
                "The unit has been flagged for further quality inspection."
            )
        else:
            status_color = "#08794e"
            status_icon = "✓"
            pill_class = "good-pill"
            pill_text = "NOT FLAGGED"
            status_message = (
                "The unit was not flagged by the model. "
                "Continue following standard quality checks."
            )

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">QUALITY ASSESSMENT</div>
            <div style="font-size:45px; margin-top:16px; color:{status_color};">
                {status_icon}
            </div>
            <div class="result-name" style="color:{status_color};">
                {prediction}
            </div>
            <span class="status-pill {pill_class}">
                {pill_text}
            </span>
            <div class="result-desc" style="margin-top:14px;">
                {status_message}
            </div>
            <div class="divider"></div>
            <div class="info-row">
                <span class="label">Defect probability</span>
                <span class="value">{probability:.2%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.caption("Estimated defect probability")
        st.progress(max(0.0, min(1.0, probability)))

        if is_defective:
            st.warning(
                "Follow the required inspection procedure before making "
                "a final quality decision."
            )
        else:
            st.info(
                "A non-defective prediction does not guarantee that the "
                "unit is free from defects."
            )

    else:
        st.markdown("""
        <div class="result-card">
            <div class="result-label">QUALITY ASSESSMENT</div>
            <div style="font-size:52px; margin-top:25px;">🔍</div>
            <div class="result-name" style="color:#213a60;">
                Ready to analyze
            </div>
            <div class="result-desc" style="margin-top:10px;">
                Complete the quality parameters and select
                <b>Predict Defect</b> to view your assessment.
            </div>
            <div class="divider"></div>
            <div class="result-desc">
                Your prediction will appear here.
            </div>
        </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# 10. Footer
# --------------------------------------------------
st.markdown("""
<div class="footer">
    AC QUALITY INTELLIGENCE &nbsp; • &nbsp; QUALITY SCREENING
    <br><br>
    AI predictions support inspection decisions and do not replace
    required physical quality checks.
</div>
""", unsafe_allow_html=True)
