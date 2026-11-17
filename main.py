
import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Smart Parking Dashboard",
    page_icon="🅿️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "parking_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "parking_data.csv")


# ---------------- LOAD MODEL AND DATA ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.title("🅿️ Smart Parking System")
st.caption("Machine Learning Based Parking Availability Dashboard")

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error("Unable to load the model or dataset.")
    st.exception(e)
    st.stop()


# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🚗 Parking Prediction")
st.sidebar.write("Enter the parking conditions below.")

day_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

available_days = df["Day"].dropna().unique().tolist()
days = sorted(
    available_days,
    key=lambda x: day_order.index(x)
    if x in day_order else len(day_order)
)

zones = sorted(df["Parking_Zone"].dropna().unique().tolist())
working_days = df["Working_Day"].dropna().unique().tolist()
weather_options = df["Weather"].dropna().unique().tolist()

with st.sidebar.form("prediction_form"):
    day = st.selectbox("Day of the Week", days)
    hour = st.selectbox("Hour of the Day", list(range(24)), index=9)
    zone = st.selectbox("Parking Zone", zones)
    working_day = st.selectbox("Working Day", working_days)
    weather = st.selectbox("Weather", weather_options)

    submitted = st.form_submit_button(
        "🔍 Predict Parking",
        use_container_width=True
    )


# ---------------- MAKE PREDICTION ----------------
if submitted:
    try:
        input_data = pd.DataFrame([{
            "Hour": hour,
            "Day": day,
            "Parking_Zone": zone,
            "Working_Day": working_day,
            "Weather": weather
        }])

        prediction = model.predict(input_data)[0]

        probabilities = {}
        if hasattr(model, "predict_proba"):
            values = model.predict_proba(input_data)[0]
            probabilities = {
                str(label): float(value)
                for label, value in zip(model.classes_, values)
            }

        st.session_state["parking_result"] = {
            "prediction": str(prediction),
            "day": day,
            "hour": hour,
            "zone": zone,
            "working_day": working_day,
            "weather": weather,
            "probabilities": probabilities
        }

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)


# ---------------- DASHBOARD SUMMARY ----------------
result = st.session_state.get("parking_result")

st.divider()
st.subheader("📊 Dashboard Overview")

total_records = len(df)
total_zones = df["Parking_Zone"].nunique()
most_common_status = df["Availability"].mode()

status_text = (
    result["prediction"] if result
    else "Not predicted yet"
)

zone_text = result["zone"] if result else "—"
time_text = (
    f'{result["hour"]:02d}:00' if result else "—"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("📁 Dataset Records", total_records)
m2.metric("🅿️ Parking Zones", total_zones)
m3.metric("🎯 Predicted Status", status_text)
m4.metric("🕒 Selected Time", time_text)

if result:
    st.subheader("🚦 Prediction Result")

    if result["prediction"] == "Available":
        st.success("🟢 Parking is predicted to be AVAILABLE.")
    elif result["prediction"] == "Limited":
        st.warning("🟡 Parking availability is predicted to be LIMITED.")
    elif result["prediction"] == "Full":
        st.error("🔴 Parking is predicted to be FULL.")
    else:
        st.info(f'Predicted status: {result["prediction"]}')

    st.write(
        f'**Selected conditions:** {result["day"]}, '
        f'{result["hour"]:02d}:00, Zone {result["zone"]}, '
        f'Working Day: {result["working_day"]}, '
        f'Weather: {result["weather"]}'
    )

    if result["probabilities"]:
        st.subheader("📈 Prediction Probabilities")

        probability_df = pd.DataFrame({
            "Probability (%)": {
                label: round(value * 100, 2)
                for label, value in result["probabilities"].items()
            }
        })

        st.bar_chart(
    probability_df,
    x_label="Parking Status",
    y_label="Probability (%)",
    height=350
)

        st.caption(
            "Probabilities are model estimates, not guarantees "
            "of actual parking availability."
        )
else:
    st.info(
        "Choose the conditions in the sidebar and click "
        "'Predict Parking' to see a prediction."
    )


# ---------------- DATASET ANALYTICS ----------------
st.divider()
st.subheader("📉 Parking Dataset Analytics")

left, right = st.columns(2)

with left:
    st.markdown("**Availability Distribution**")

    status_counts = (
        df["Availability"]
        .value_counts()
        .reindex(["Available", "Limited", "Full"], fill_value=0)
    )

    status_chart = pd.DataFrame({
        "Number of Records": status_counts
    })

    st.bar_chart(
    status_chart,
    x_label="Availability Status",
    y_label="Number of Records",
    height=350
)

with right:
    st.markdown("**Average Future Occupancy by Zone**")

    occupancy_by_zone = (
        df.groupby("Parking_Zone")["Future_Occupancy"]
        .mean()
        .sort_index()
    )

    occupancy_chart = pd.DataFrame({
        "Average Occupancy": occupancy_by_zone
    })

    st.bar_chart(
    occupancy_chart,
    x_label="Parking Zone",
    y_label="Average Future Occupancy",
    height=350
)

st.divider()

st.subheader("📋 Dataset Preview")
with st.expander("View parking dataset"):
    st.dataframe(df, use_container_width=True)

st.caption(
    "Smart Parking System | Streamlit | Decision Tree Classifier"
)
