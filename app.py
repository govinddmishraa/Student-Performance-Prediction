import streamlit as st
import joblib
import pandas as pd
import json
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODEL + METRICS + DATASET
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/student_performance_model.pkl"
    )


@st.cache_data
def load_metrics():
    with open("models/metrics.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    return pd.read_csv(
        "data/StudentPerformanceFactors.csv"
    )


model = load_model()
metrics = load_metrics()
dataset = load_dataset()


# ============================================================
# DATA VALIDATION FOR DISPLAY
# ============================================================

valid_dataset = dataset[
    dataset["Exam_Score"].between(0, 100)
].copy()


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = max(
    metrics,
    key=lambda name: metrics[name]["R2"]
)

best_metrics = metrics[best_model_name]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 EduPredict AI")

st.sidebar.write(
    "AI-Based Student Performance Prediction System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Prediction",
        "📊 Data Analysis",
        "🧠 Model Performance",
        "🔍 Feature Insights",
        "🏗️ Architecture",
        "ℹ️ About"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    f"Selected Model: {best_model_name}"
)


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "🏠 Prediction":

    st.title("🎓 EduPredict AI")

    st.subheader(
        "AI-Based Student Performance Prediction System"
    )

    st.write(
        "Enter a student's academic and background information "
        "to estimate their expected exam score."
    )

    st.divider()


    # --------------------------------------------------------
    # ACADEMIC INFORMATION
    # --------------------------------------------------------

    st.header("📊 Student Academic Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        hours_studied = st.number_input(
            "Hours Studied",
            min_value=1,
            max_value=44,
            value=20
        )

    with col2:

        attendance = st.number_input(
            "Attendance (%)",
            min_value=60,
            max_value=100,
            value=85
        )

    with col3:

        sleep_hours = st.number_input(
            "Sleep Hours",
            min_value=4,
            max_value=10,
            value=7
        )


    col4, col5, col6 = st.columns(3)

    with col4:

        previous_scores = st.number_input(
            "Previous Scores",
            min_value=50,
            max_value=100,
            value=75
        )

    with col5:

        tutoring_sessions = st.number_input(
            "Tutoring Sessions",
            min_value=0,
            max_value=8,
            value=2
        )

    with col6:

        physical_activity = st.number_input(
            "Physical Activity (hours/week)",
            min_value=0,
            max_value=6,
            value=3
        )


    # --------------------------------------------------------
    # BACKGROUND INFORMATION
    # --------------------------------------------------------

    st.header("👨‍🎓 Student & Background Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        parental_involvement = st.selectbox(
            "Parental Involvement",
            ["Low", "Medium", "High"]
        )

    with col2:

        access_to_resources = st.selectbox(
            "Access to Resources",
            ["Low", "Medium", "High"]
        )

    with col3:

        extracurricular_activities = st.selectbox(
            "Extracurricular Activities",
            ["No", "Yes"]
        )


    col4, col5, col6 = st.columns(3)

    with col4:

        motivation_level = st.selectbox(
            "Motivation Level",
            ["Low", "Medium", "High"]
        )

    with col5:

        internet_access = st.selectbox(
            "Internet Access",
            ["No", "Yes"]
        )

    with col6:

        family_income = st.selectbox(
            "Family Income",
            ["Low", "Medium", "High"]
        )


    col7, col8, col9 = st.columns(3)

    with col7:

        teacher_quality = st.selectbox(
            "Teacher Quality",
            ["Low", "Medium", "High"]
        )

    with col8:

        school_type = st.selectbox(
            "School Type",
            ["Public", "Private"]
        )

    with col9:

        peer_influence = st.selectbox(
            "Peer Influence",
            ["Negative", "Neutral", "Positive"]
        )


    col10, col11, col12 = st.columns(3)

    with col10:

        learning_disabilities = st.selectbox(
            "Learning Disabilities",
            ["No", "Yes"]
        )

    with col11:

        parental_education = st.selectbox(
            "Parental Education Level",
            [
                "High School",
                "College",
                "Postgraduate"
            ]
        )

    with col12:

        distance_from_home = st.selectbox(
            "Distance from Home",
            [
                "Near",
                "Moderate",
                "Far"
            ]
        )


    col13, _, _ = st.columns(3)

    with col13:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )


    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "Hours_Studied": [hours_studied],

        "Attendance": [attendance],

        "Parental_Involvement": [
            parental_involvement
        ],

        "Access_to_Resources": [
            access_to_resources
        ],

        "Extracurricular_Activities": [
            extracurricular_activities
        ],

        "Sleep_Hours": [sleep_hours],

        "Previous_Scores": [
            previous_scores
        ],

        "Motivation_Level": [
            motivation_level
        ],

        "Internet_Access": [
            internet_access
        ],

        "Tutoring_Sessions": [
            tutoring_sessions
        ],

        "Family_Income": [
            family_income
        ],

        "Teacher_Quality": [
            teacher_quality
        ],

        "School_Type": [
            school_type
        ],

        "Peer_Influence": [
            peer_influence
        ],

        "Physical_Activity": [
            physical_activity
        ],

        "Learning_Disabilities": [
            learning_disabilities
        ],

        "Parental_Education_Level": [
            parental_education
        ],

        "Distance_from_Home": [
            distance_from_home
        ],

        "Gender": [
            gender
        ]
    })


    # --------------------------------------------------------
    # INPUT PREVIEW
    # --------------------------------------------------------

    with st.expander(
        "🔍 View Model Input Data"
    ):

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🎯 Predict Exam Score",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        # Actual ML prediction
        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.header("📈 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Predicted Exam Score",
                f"{prediction:.2f} / 100"
            )

        with result_col2:

            if prediction >= 80:

                performance = "Excellent"

            elif prediction >= 70:

                performance = "Good"

            elif prediction >= 60:

                performance = "Average"

            else:

                performance = "Needs Improvement"


            st.metric(
                "Performance Level",
                performance
            )


        st.caption(
            "The predicted score is a machine-learning "
            "estimate and is not a guaranteed result."
        )


        # ----------------------------------------------------
        # PERSONALIZED INSIGHTS
        # ----------------------------------------------------

        st.subheader(
            "💡 Personalized Student Insights"
        )

        recommendations = []


        if attendance < 75:

            recommendations.append(
                "📅 Attendance is relatively low. "
                "Consider improving regular class participation."
            )


        if hours_studied < 15:

            recommendations.append(
                "📚 Consider increasing study time and "
                "following a consistent study schedule."
            )


        if previous_scores < 65:

            recommendations.append(
                "📝 Focus on revision and strengthening "
                "fundamental concepts."
            )


        if tutoring_sessions == 0:

            recommendations.append(
                "👨‍🏫 Consider academic support or tutoring "
                "if additional subject assistance is needed."
            )


        if sleep_hours < 6:

            recommendations.append(
                "😴 Consider maintaining a healthier and "
                "more consistent sleep routine."
            )


        if physical_activity == 0:

            recommendations.append(
                "🏃 Consider including some regular "
                "physical activity in your routine."
            )


        if not recommendations:

            recommendations.append(
                "✅ No specific improvement suggestions "
                "were triggered by the entered profile."
            )


        for recommendation in recommendations:

            st.write(
                recommendation
            )


# ============================================================
# PAGE 2 — DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    st.title("📊 Dataset Analysis")

    st.write(
        "Overview and exploratory analysis of the "
        "Student Performance Factors dataset."
    )


    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Original Records",
            f"{len(dataset):,}"
        )

    with col2:

        st.metric(
            "Valid Records",
            f"{len(valid_dataset):,}"
        )

    with col3:

        st.metric(
            "Input Features",
            "19"
        )

    with col4:

        st.metric(
            "Target",
            "Exam_Score"
        )


    st.divider()


    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📈 Exam Score Distribution"
    )

    score_distribution = (
        valid_dataset[
            "Exam_Score"
        ]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        score_distribution
    )


    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader(
        "🔗 Numeric Feature Correlation"
    )

    numeric_columns = valid_dataset.select_dtypes(
        include="number"
    ).columns

    correlation = (
        valid_dataset[numeric_columns]
        .corr()["Exam_Score"]
        .sort_values(
            ascending=False
        )
    )

    st.dataframe(
        correlation.to_frame(
            "Correlation with Exam Score"
        ),
        use_container_width=True
    )


    # --------------------------------------------------------
    # HOURS STUDIED
    # --------------------------------------------------------

    st.subheader(
        "📚 Hours Studied vs Exam Score"
    )

    hours_analysis = (
        valid_dataset
        .groupby("Hours_Studied")[
            "Exam_Score"
        ]
        .mean()
    )

    st.line_chart(
        hours_analysis
    )


    # --------------------------------------------------------
    # ATTENDANCE
    # --------------------------------------------------------

    st.subheader(
        "📅 Attendance vs Exam Score"
    )

    attendance_analysis = (
        valid_dataset
        .groupby("Attendance")[
            "Exam_Score"
        ]
        .mean()
    )

    st.line_chart(
        attendance_analysis
    )


    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    st.subheader(
        "🔎 Data Quality"
    )

    missing_values = (
        dataset.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_values = missing_values[
        missing_values > 0
    ]

    if len(missing_values) > 0:

        st.dataframe(
            missing_values.to_frame(
                "Missing Values"
            ),
            use_container_width=True
        )

    else:

        st.success(
            "No missing values found."
        )


    st.write(
        f"Duplicate rows: "
        f"**{dataset.duplicated().sum()}**"
    )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "🧠 Model Performance":

    st.title("🧠 Model Performance")

    st.write(
        "Comparison of the regression models evaluated "
        "during training."
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    st.success(
        f"🏆 Selected Model: {best_model_name}"
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"{best_metrics['MAE']:.3f}"
        )

    with col2:

        st.metric(
            "RMSE",
            f"{best_metrics['RMSE']:.3f}"
        )

    with col3:

        st.metric(
            "R² Score",
            f"{best_metrics['R2']:.3f}"
        )


    st.divider()


    # --------------------------------------------------------
    # MODEL COMPARISON TABLE
    # --------------------------------------------------------

    st.subheader(
        "📊 Model Comparison"
    )

    comparison_data = []

    for name, values in metrics.items():

        comparison_data.append({

            "Model": name,

            "MAE": round(
                values["MAE"],
                3
            ),

            "RMSE": round(
                values["RMSE"],
                3
            ),

            "R²": round(
                values["R2"],
                3
            )
        })


    comparison_df = pd.DataFrame(
        comparison_data
    )


    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # R2 CHART
    # --------------------------------------------------------

    st.subheader(
        "📈 R² Score Comparison"
    )

    r2_chart = (
        comparison_df
        .set_index("Model")[["R²"]]
    )

    st.bar_chart(
        r2_chart
    )


    # --------------------------------------------------------
    # METRIC EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "📖 Understanding the Metrics"
    )

    st.write(
        "**MAE:** Average absolute difference between "
        "actual and predicted scores."
    )

    st.write(
        "**RMSE:** Measures prediction error while "
        "giving greater weight to larger errors."
    )

    st.write(
        "**R²:** Indicates how much of the variation "
        "in the target is explained by the model."
    )


# ============================================================
# PAGE 4 — FEATURE INSIGHTS
# ============================================================

elif page == "🔍 Feature Insights":

    st.title("🔍 Feature Insights")

    st.write(
        "Descriptive relationships between numeric "
        "features and the target variable."
    )


    # --------------------------------------------------------
    # CORRELATION TABLE
    # --------------------------------------------------------

    numeric_columns = valid_dataset.select_dtypes(
        include="number"
    ).columns

    correlations = (
        valid_dataset[numeric_columns]
        .corr()["Exam_Score"]
        .drop("Exam_Score")
        .sort_values(
            ascending=False
        )
    )


    st.subheader(
        "📌 Correlation with Exam Score"
    )

    feature_df = correlations.to_frame(
        "Correlation"
    )

    st.dataframe(
        feature_df,
        use_container_width=True
    )


    # --------------------------------------------------------
    # STRONGEST POSITIVE NUMERIC RELATIONSHIP
    # --------------------------------------------------------

    strongest_positive = correlations.idxmax()

    strongest_positive_value = correlations.max()


    st.info(
        f"Among the numeric variables, "
        f"**{strongest_positive}** has the strongest "
        f"positive correlation with Exam Score "
        f"({strongest_positive_value:.3f})."
    )


    st.warning(
        "Correlation describes association in the dataset; "
        "it does not establish causation or guarantee that "
        "changing a feature will produce the same change "
        "in exam score."
    )


    # --------------------------------------------------------
    # CATEGORICAL GROUP ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "📊 Categorical Feature Analysis"
    )


    categorical_columns = [
        column
        for column in valid_dataset.columns
        if valid_dataset[column].dtype == "object"
    ]


    selected_feature = st.selectbox(
        "Select a categorical feature",
        categorical_columns
    )


    category_analysis = (
        valid_dataset
        .groupby(selected_feature)[
            "Exam_Score"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )


    st.bar_chart(
        category_analysis
    )


    st.dataframe(
        category_analysis.to_frame(
            "Average Exam Score"
        ),
        use_container_width=True
    )


# ============================================================
# PAGE 5 — ARCHITECTURE
# ============================================================

elif page == "🏗️ Architecture":

    st.title("🏗️ System Architecture")

    st.write(
        "End-to-end workflow of the AI-based student "
        "performance prediction system."
    )


    st.code(
        """
                    STUDENT DATASET
                          │
                          ▼
                  DATA VALIDATION
                          │
                          ▼
                 FEATURE PROCESSING
              ┌───────────┴───────────┐
              │                       │
        Numeric Features       Categorical Features
              │                       │
        Median Imputation       Most-Frequent
              │                    Imputation
        Standard Scaling              │
              │                 One-Hot Encoding
              └───────────┬───────────┘
                          │
                          ▼
                   TRAIN / TEST SPLIT
                          │
                          ▼
                ┌─────────┼─────────┐
                │         │         │
                ▼         ▼         ▼
             Linear    Random    Gradient
            Regression  Forest    Boosting
                │         │         │
                └─────────┼─────────┘
                          │
                          ▼
                  MODEL EVALUATION
                  MAE / RMSE / R²
                          │
                          ▼
                   BEST MODEL
                Linear Regression
                          │
                          ▼
              student_performance_model.pkl
                          │
                          ▼
                    STREAMLIT APP
                          │
                          ▼
                  STUDENT INPUTS
                          │
                          ▼
                    PREDICTION
                          │
                          ▼
                 PERSONALIZED INSIGHTS
        """,
        language="text"
    )


    st.subheader(
        "🔄 Prediction Pipeline"
    )

    st.write(
        "1. User enters the 19 student features."
    )

    st.write(
        "2. The inputs are converted into a Pandas DataFrame."
    )

    st.write(
        "3. The saved Pipeline performs preprocessing."
    )

    st.write(
        "4. The trained Linear Regression model generates "
        "the predicted Exam Score."
    )

    st.write(
        "5. The application displays the predicted score "
        "and rule-based student insights."
    )


# ============================================================
# PAGE 6 — ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About EduPredict AI")

    st.subheader(
        "Project Objective"
    )

    st.write(
        "EduPredict AI is an AI-based student performance "
        "prediction system designed to estimate a student's "
        "expected exam score from academic and background "
        "features."
    )


    st.subheader(
        "🎯 Machine Learning Task"
    )

    st.write(
        "This project is a supervised regression problem "
        "because the target variable, Exam_Score, is numerical."
    )


    st.subheader(
        "🤖 Algorithms Evaluated"
    )

    st.write(
        "• Linear Regression"
    )

    st.write(
        "• Random Forest Regressor"
    )

    st.write(
        "• Gradient Boosting Regressor"
    )


    st.subheader(
        "⚙️ Preprocessing"
    )

    st.write(
        "• Median imputation for numerical missing values"
    )

    st.write(
        "• Most-frequent imputation for categorical values"
    )

    st.write(
        "• StandardScaler for numerical features"
    )

    st.write(
        "• OneHotEncoder for categorical features"
    )


    st.subheader(
        "🏆 Final Model"
    )

    st.success(
        f"{best_model_name}"
    )


    st.subheader(
        "📊 Final Test Performance"
    )

    st.write(
        f"MAE: **{best_metrics['MAE']:.3f}**"
    )

    st.write(
        f"RMSE: **{best_metrics['RMSE']:.3f}**"
    )

    st.write(
        f"R²: **{best_metrics['R2']:.3f}**"
    )


    st.subheader(
        "⚠️ Important Limitation"
    )

    st.warning(
        "The dataset used for this project is synthetic and "
        "the model's predictions should be treated as estimates, "
        "not guaranteed academic outcomes."
    )


    st.caption(
        "EduPredict AI — Student Performance Prediction System"
    )