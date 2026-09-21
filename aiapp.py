import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Hiring Bias Detection",
    page_icon="🤖",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv("AI_Hiring_Bias_Dataset.csv")


df = load_data()


st.title("🤖 AI Hiring Bias Detection")
st.subheader("Machine Learning & Fairness Analysis")

st.write(
    """
    This dashboard presents an analysis of simulated AI-assisted
    hiring outcomes using exploratory data analysis, machine learning,
    and fairness metrics.
    """
)

st.sidebar.title("Navigation🔍")

page = st.sidebar.radio(
    "Go to-",
    [
    "Overview",
    "Dataset Analysis",
    "Fairness Analysis",
    "ML Model Results",
    "Feature Importance",
    "Methodology",
    "Conclusion",
]
)

if page == "Overview":

    st.header("📊 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Candidates",
            len(df)
        )

    with col2:
        st.metric(
            "Hiring Rate",
            f"{df['hired'].mean() * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Average AI Score",
            f"{df['ai_resume_score'].mean():.2f}"
        )

    with col4:
        st.metric(
            "Average AI Bias Score",
            f"{df['ai_bias_score'].mean():.2f}"
        )

    st.markdown("---")

    st.write("### Research Question")

    st.info(
        """
        Does the simulated AI hiring system produce systematically
        different evaluations or decisions across candidate groups,
        and which candidate characteristics are associated with
        those outcomes?
        """
    )

    st.write("### Project Components")

    st.write(
        """
        • Exploratory Data Analysis  
        • AI score analysis  
        • Gender-wise hiring analysis  
        • Fairness metrics  
        • Logistic Regression  
        • Random Forest  
        • Feature importance  
        • Statistical analysis
        """
    )
    

if page == "Dataset Analysis":

    st.header("📊 Dataset Analysis")

    st.subheader("Hiring Distribution")

    hiring_counts = df["hired"].value_counts()

    st.bar_chart(hiring_counts)

    st.subheader("Hiring Rate by Gender")

    hiring_gender = (
        df.groupby("gender")["hired"]
        .mean()
        * 100
    )

    st.bar_chart(hiring_gender)

    st.subheader("Average AI Resume Score by Gender")

    ai_gender = (
        df.groupby("gender")["ai_resume_score"]
        .mean()
    )

    st.bar_chart(ai_gender)

    st.subheader("Average AI Bias Score by Gender")

    bias_gender = (
        df.groupby("gender")["ai_bias_score"]
        .mean()
    )

    st.bar_chart(bias_gender)



if page == "Fairness Analysis":

    st.header("⚖️ Fairness Analysis")

    st.write(
        """
        This section examines whether hiring outcomes and AI-related
        scores differ across gender groups.
        """
    )

    
    gender_options = ["All"] + sorted(
        df["gender"].dropna().unique().tolist()
    )

    selected_gender = st.selectbox(
        "Select Gender",
        gender_options
    )


    if selected_gender == "All":
        fairness_df = df.copy()
    else:
        fairness_df = df[
            df["gender"] == selected_gender
        ]

    st.write(
        f"Number of candidates analyzed: {len(fairness_df)}"
    )

    
    col1, col2, col3 = st.columns(3)

    with col1:
        hiring_rate = fairness_df["hired"].mean() * 100

        st.metric(
            "Hiring Rate",
            f"{hiring_rate:.2f}%"
        )

    with col2:
        average_ai_score = fairness_df[
            "ai_resume_score"
        ].mean()

        st.metric(
            "Average AI Score",
            f"{average_ai_score:.2f}"
        )

    with col3:
        average_bias_score = fairness_df[
            "ai_bias_score"
        ].mean()

        st.metric(
            "Average AI Bias Score",
            f"{average_bias_score:.2f}"
        )

    st.markdown("---")

    
    if selected_gender == "All":

        st.subheader("Hiring Rate by Gender")

        selection_rate = (
            df.groupby("gender")["hired"]
            .mean() * 100
        )

        st.bar_chart(selection_rate)

        st.subheader("Average AI Resume Score by Gender")

        ai_score_gender = (
            df.groupby("gender")["ai_resume_score"]
            .mean()
        )

        st.bar_chart(ai_score_gender)

        st.subheader("Average AI Bias Score by Gender")

        ai_bias_gender = (
            df.groupby("gender")["ai_bias_score"]
            .mean()
        )

        st.bar_chart(ai_bias_gender)

    else:

        st.subheader(
            f"Analysis for {selected_gender}"
        )

        st.write(
            fairness_df[
                [
                    "age",
                    "education_level",
                    "years_experience",
                    "ai_resume_score",
                    "ai_bias_score",
                    "hired"
                ]
            ]
        )

    st.warning(
        """
        These metrics describe differences between groups.
        They do not by themselves establish the cause of those
        differences or prove discrimination.
        """
    )


    fairness_csv = (
     fairness_df.to_csv(index=False)
)

    st.download_button(
    label="📥 Download Fairness Data",
    data=fairness_csv,
    file_name="Fairness_Analysis.csv",
    mime="text/csv"
)
    

if page == "ML Model Results":

    st.header("🤖 Machine Learning Results")

    ml_results = pd.read_csv(
        "results/Final_ML_Summary.csv"
    )

    st.subheader("Model Performance")

    
    selected_model = st.selectbox(
        "Select a model",
        ml_results["Model"].tolist()
    )

    selected_result = ml_results[
        ml_results["Model"] == selected_model
    ].iloc[0]

    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{selected_result['Accuracy']:.3f}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{selected_result['Precision']:.3f}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{selected_result['Recall']:.3f}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{selected_result['F1 Score']:.3f}"
        )

    with col5:
        st.metric(
            "ROC-AUC",
            f"{selected_result['ROC-AUC']:.3f}"
        )

    st.markdown("---")

    st.subheader("Complete Model Comparison")

    st.dataframe(
        ml_results,
        use_container_width=True
    )
      # Download ML results
      
    csv_data = ml_results.to_csv(
          index=False
      )
      
    st.download_button(
          label="📥 Download ML Results",
          data=csv_data,
          file_name="Final_ML_Summary.csv",
          mime="text/csv"
      )
      
    st.subheader("Model Performance Chart")

    st.bar_chart(
        ml_results.set_index("Model")[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ]
        ]
    )
# Feature Importance page

if page == "Feature Importance":

    st.header("🔍 Random Forest Feature Importance")

    feature_importance = pd.read_csv(
        "results/Random_Forest_Feature_Importance.csv"
    )

    feature_importance = feature_importance.sort_values(
        "Importance",
        ascending=False
    )

    st.subheader("Top 15 Features")

    top_features = feature_importance.head(15)

    st.bar_chart(
        top_features.set_index("Feature")[
            "Importance"
        ]
    )

    st.dataframe(
        top_features,
        use_container_width=True
    )

st.sidebar.markdown("---")

st.sidebar.write(
    """
    ### About

    AI Hiring Bias Detection

    Machine Learning College Project

    Techniques:
    • EDA
    • Fairness Analysis
    • Logistic Regression
    • Random Forest
    • Statistical Analysis
    """
    
)
# Methodology page

if page == "Methodology":

    st.header("🔬 Project Methodology")

    st.subheader("1. Data Collection")

    st.write(
        """
        The project uses a simulated AI hiring dataset containing
        candidate demographic information, qualifications, AI-related
        scores, and hiring outcomes.
        """
    )

    st.subheader("2. Data Analysis")

    st.write(
        """
        Exploratory Data Analysis was performed to understand the
        distribution of candidate characteristics, AI scores, and
        hiring outcomes.
        """
    )

    st.subheader("3. Fairness Analysis")

    st.write(
        """
        Hiring outcomes and AI-related scores were compared across
        gender groups. Demographic parity difference and demographic
        parity ratio were used as group-level fairness measures.
        """
    )

    st.subheader("4. Machine Learning")

    st.write(
        """
        Two classification models were trained:

        • Logistic Regression
        • Random Forest
        """
    )

    st.subheader("5. Statistical Analysis")

    st.write(
        """
        One-way ANOVA was used to examine whether mean AI resume
        scores differed across gender groups.
        """
    )

    st.subheader("6. Evaluation")

    st.write(
        """
        Models were evaluated using:

        • Accuracy
        • Precision
        • Recall
        • F1 Score
        • ROC-AUC
        • Confusion Matrix
        """
    )
# Conclusion page

if page == "Conclusion":

    st.header("📝 Project Conclusion")

    st.subheader("Key Findings")

    st.write(
        """
        The analysis examined differences in hiring outcomes,
        AI resume scores, and AI bias scores across candidate groups.
        Machine learning models were also used to identify candidate
        characteristics associated with the recorded hiring outcomes.
        """
    )

    st.subheader("Fairness Analysis")

    st.write(
        """
        Differences in selection rates and AI-related scores between
        groups were measured using fairness metrics. These measurements
        describe observed disparities but do not by themselves establish
        their cause or prove discrimination.
        """
    )

    st.subheader("Machine Learning")

    st.write(
        """
        Logistic Regression and Random Forest were trained to model
        the recorded hiring outcomes. Their performance was evaluated
        using standard classification metrics.
        """
    )

    st.subheader("Overall Interpretation")

    st.info(
        """
        This project demonstrates how machine learning and fairness
        analysis can be combined to audit simulated AI-assisted hiring
        outcomes and identify patterns that may require further
        investigation.
        """
    )

    st.subheader("Important Limitation")

    st.warning(
        """
        The dataset is simulated and the analysis does not represent
        the hiring practices of a real organization. Observed
        statistical differences should therefore be interpreted as
        dataset-level findings rather than proof of real-world
        discrimination.
        """
    )
