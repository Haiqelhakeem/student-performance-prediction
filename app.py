import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

### Streamlit App
st.set_page_config(layout="wide", page_title="Student Dropout Prediction - Jaya Jaya Institut")
st.title("Student Dropout Prediction - Jaya Jaya Institut")

# Optimized caching for model loading
@st.cache_resource
def load_model(filename):
    with open(filename, 'rb') as f:
        return joblib.load(f)

rf_model = load_model('dropout_rf_model.pkl')
xgb_model = load_model('dropout_xgb_model.pkl')
ordinal_encoder = load_model('ordinal_encoder.pkl')
label_encoder = load_model('label_encoder.pkl')

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data.csv", sep=";")

data = load_data()
df = data.copy()

# Apply scaling only for visualization
df['Admission_grade_scaled'] = df['Admission_grade'] / 10

# Top features
top_features = [
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_approved',
    'Tuition_fees_up_to_date',
    'Age_at_enrollment',
    'Scholarship_holder',
    'Debtor',
    'Gender',
    'Application_mode'
]

reverse_gender = {1: "Male", 0: "Female"}
reverse_binary = {1: "Yes", 0: "No"}
gender_mapping = {"Male": 1, "Female": 0}

application_mode_mapping = {
    "1st phase - general contingent": 1, "Ordinance No. 612/93": 2,
    "1st phase - special contingent (Azores Island)": 5, "Holders of other higher courses": 7,
    "Ordinance No. 854-B/99": 10, "International student (bachelor)": 15,
    "1st phase - special contingent (Madeira Island)": 16, "2nd phase - general contingent": 17,
    "3rd phase - general contingent": 18, "Ordinance No. 533-A/99 b2": 26,
    "Ordinance No. 533-A/99 b3": 27, "Over 23 years old": 39, "Transfer": 42,
    "Change of course": 43, "Technological specialization diploma holders": 44,
    "Change of institution/course": 51, "Short cycle diploma holders": 53,
    "Change of institution/course (International)": 57
}

@st.cache_data
def compute_corr(df, label_encoder):
    df = df.copy()
    df['Status_encoded'] = label_encoder.transform(df['Status'])
    corr = df.corr(numeric_only=True)
    return corr

@st.cache_data
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] < lower) | (data[column] > upper)]

# Sidebar
model_choice = st.sidebar.radio("Choose Model", ["Random Forest", "XGBoost"])
tabs = st.tabs(["Data Visualization", "Prediction", "Recommendations"])

# --- Data Visualization Tab ---
with tabs[0]:
    st.header("Data Visualization")

    viz_tabs = st.tabs(["Overview", "Demographics", "Financial Factors", "Academic Factors"])

    with viz_tabs[0]:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        st.subheader("Class Distribution")
        st.bar_chart(df['Status'].value_counts())

        with st.expander("Correlation Heatmap (Top Features Only)"):
            data['Status_encoded'] = label_encoder.transform(data['Status'])
            corr = data[top_features + ['Status_encoded']].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(fig)

        with st.expander("Top 10 Correlated Features"):
            top_corr = corr['Status_encoded'].abs().sort_values(ascending=False)[1:11]
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x=top_corr.values, y=top_corr.index, palette="magma", ax=ax)
            plt.xlabel("Correlation with Dropout")
            st.pyplot(fig)

    with viz_tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dropout by Gender")
            df['Gender_str'] = df['Gender'].map({0: 'Female', 1: 'Male'})
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(data=df, x='Gender_str', hue='Status', ax=ax)
            st.pyplot(fig)
            for status in df['Status'].unique():
                count_male = len(df[(df['Gender'] == 1) & (df['Status'] == status)])
                count_female = len(df[(df['Gender'] == 0) & (df['Status'] == status)])
                st.write(f"{status} - Male: {count_male}, Female: {count_female}")

        with col2:
            st.subheader("Dropout by Age Group")
            df['Age_Group'] = pd.cut(df['Age_at_enrollment'], bins=[15,20,25,30,35,40,100],
                                     labels=["<20", "20-24", "25-29", "30-34", "35-39", "40+"])
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(data=df, x='Age_Group', hue='Status', ax=ax)
            st.pyplot(fig)
            for status in df['Status'].unique():
                for group in df['Age_Group'].unique():
                    count = len(df[(df['Age_Group'] == group) & (df['Status'] == status)])
                    st.write(f"{status} - {group}: {count}")

    with viz_tabs[2]:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dropout by Scholarship Holder")
            df['Scholarship_str'] = df['Scholarship_holder'].map({0: 'No', 1: 'Yes'})
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(data=df, x='Scholarship_str', hue='Status', ax=ax)
            st.pyplot(fig)
            for status in df['Status'].unique():
                for cat in ['No', 'Yes']:
                    count = len(df[(df['Scholarship_str'] == cat) & (df['Status'] == status)])
                    st.write(f"{status} - Scholarship {cat}: {count}")

        with col2:
            st.subheader("Dropout by Debtor")
            df['Debtor_str'] = df['Debtor'].map({0: 'No', 1: 'Yes'})
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(data=df, x='Debtor_str', hue='Status', ax=ax)
            st.pyplot(fig)
            for status in df['Status'].unique():
                for cat in ['No', 'Yes']:
                    count = len(df[(df['Debtor_str'] == cat) & (df['Status'] == status)])
                    st.write(f"{status} - Debtor {cat}: {count}")

        st.subheader("Dropout by Tuition Fees Payment Status")
        df['Tuition_str'] = df['Tuition_fees_up_to_date'].map({0: 'No', 1: 'Yes'})
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(data=df, x='Tuition_str', hue='Status', ax=ax)
        st.pyplot(fig)
        for status in df['Status'].unique():
            for cat in ['No', 'Yes']:
                count = len(df[(df['Tuition_str'] == cat) & (df['Status'] == status)])
                st.write(f"{status} - Tuition Up To Date {cat}: {count}")

    with viz_tabs[3]:
        st.subheader("Feature Distributions")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df['Admission_grade_scaled'], kde=True, ax=ax)
            ax.set_title("Admission Grade")
            st.pyplot(fig)
            st.write(f"Mean: {df['Admission_grade_scaled'].mean():.2f}")
            st.write(f"Median: {df['Admission_grade_scaled'].median():.2f}")
            st.write(f"Min: {df['Admission_grade_scaled'].min():.2f}")
            st.write(f"Max: {df['Admission_grade_scaled'].max():.2f}")

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df['Age_at_enrollment'], kde=True, ax=ax)
            ax.set_title("Age at Enrollment")
            st.pyplot(fig)
            st.write(f"Mean: {df['Age_at_enrollment'].mean():.2f}")
            st.write(f"Median: {df['Age_at_enrollment'].median():.2f}")
            st.write(f"Min: {df['Age_at_enrollment'].min():.0f}")
            st.write(f"Max: {df['Age_at_enrollment'].max():.0f}")

        with st.expander("Detailed Boxplots of Top Correlated Features"):
            most_corr_features = corr['Status_encoded'].sort_values(key=abs, ascending=False).index[1:11]
            for col in most_corr_features:
                st.write(f"Boxplot for {col}")
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)
                outliers = detect_outliers_iqr(df, col)
                st.write(f"Total Outliers in {col}: {len(outliers)}")


# --- Prediction Tab ---
with tabs[1]:
    st.header("🎯 Predict Dropout")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            Curricular_units_2nd_sem_grade = st.number_input("2nd Sem Grade", 0.0, 20.0, step=0.1)
            Curricular_units_2nd_sem_approved = st.number_input("2nd Sem Approved", 0, 100)
            Curricular_units_1st_sem_grade = st.number_input("1st Sem Grade", 0.0, 20.0, step=0.1)
            Curricular_units_1st_sem_approved = st.number_input("1st Sem Approved", 0, 100)
            Tuition_fees_up_to_date = st.selectbox("Tuition Fees Up To Date?", ["Yes", "No"])
            Age_at_enrollment = st.number_input("Age at Enrollment", 17, 65)
        with col2:
            Scholarship_holder = st.selectbox("Scholarship Holder?", ["Yes", "No"])
            Debtor = st.selectbox("Debtor?", ["Yes", "No"])
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Application_mode = st.selectbox("Application Mode", list(application_mode_mapping.keys()))
        submit = st.form_submit_button("Predict")

    if submit:
        input_dict = {
            'Curricular_units_2nd_sem_grade': Curricular_units_2nd_sem_grade,
            'Curricular_units_2nd_sem_approved': Curricular_units_2nd_sem_approved,
            'Curricular_units_1st_sem_grade': Curricular_units_1st_sem_grade,
            'Curricular_units_1st_sem_approved': Curricular_units_1st_sem_approved,
            'Tuition_fees_up_to_date': 1 if Tuition_fees_up_to_date == "Yes" else 0,
            'Age_at_enrollment': Age_at_enrollment,
            'Scholarship_holder': 1 if Scholarship_holder == "Yes" else 0,
            'Debtor': 1 if Debtor == "Yes" else 0,
            'Gender': gender_mapping[Gender],
            'Application_mode': application_mode_mapping[Application_mode]
        }

        input_df = pd.DataFrame([input_dict])
        cat_features = ['Tuition_fees_up_to_date', 'Scholarship_holder', 'Debtor', 'Gender', 'Application_mode']
        num_features = [col for col in input_df.columns if col not in cat_features]
        encoded_cats = ordinal_encoder.transform(input_df[cat_features])
        final_input = pd.concat([
            input_df[num_features].reset_index(drop=True),
            pd.DataFrame(encoded_cats, columns=cat_features)
        ], axis=1)[input_df.columns]

        if model_choice == "Random Forest":
            prediction = rf_model.predict(final_input)
            proba = rf_model.predict_proba(final_input).max() * 100
        else:
            prediction = xgb_model.predict(final_input)
            proba = xgb_model.predict_proba(final_input).max() * 100

        final_label = label_encoder.inverse_transform(prediction)[0]
        rec_action = "🛑 Immediate Intervention" if final_label == "Dropout" else ("📖 Continuous Monitoring" if final_label == "Enrolled" else "✅ No Immediate Action")
        result_df = pd.DataFrame({
            "Predicted Status": [final_label],
            "Confidence": [f"{proba:.2f}%"],
            "Recommended Action": [rec_action]
        })
        st.table(result_df)

# --- Recommendations Tab ---
with tabs[2]:
    st.header("📋 Recommendations Dashboard")

    # Apply full model prediction to entire dataset
    full_X = data[top_features].copy()
    cat_features = ['Tuition_fees_up_to_date', 'Scholarship_holder', 'Debtor', 'Gender', 'Application_mode']
    full_X[cat_features] = ordinal_encoder.transform(full_X[cat_features])

    if model_choice == "Random Forest":
        preds = rf_model.predict(full_X)
    else:
        preds = xgb_model.predict(full_X)

    data['Predicted_Status'] = label_encoder.inverse_transform(preds)

    # Map back to string for filters and display
    data['Gender_str'] = data['Gender'].map(reverse_gender)
    data['Scholarship_str'] = data['Scholarship_holder'].map(reverse_binary)
    data['Debtor_str'] = data['Debtor'].map(reverse_binary)
    data['Tuition_str'] = data['Tuition_fees_up_to_date'].map(reverse_binary)

    # Dynamic Recommended Actions Logic
    def dynamic_recommendation(row):
        if row['Predicted_Status'] == "Dropout" and row['Status'] == "Enrolled":
            return "🧑\u200d🏫 Academic Mentor, 🧠 Counseling, 💰 Financial Aid"
        elif row['Predicted_Status'] == "Dropout" and row['Status'] == "Dropout":
            return "📞 Alumni Follow-up, 🔁 Re-entry Program"
        elif row['Predicted_Status'] == "Enrolled":
            return "📖 Continuous Monitoring"
        elif row['Predicted_Status'] == "Graduate":
            return "✅ Maintain Progress"
        else:
            return "🔎 Review Case"

    data['Recommended Action'] = data.apply(dynamic_recommendation, axis=1)

    # Filters with 'Both' option
    with st.expander("🔎 Filter Options"):
        gender_options = ["Both"] + sorted(data['Gender_str'].unique().tolist())
        scholarship_options = ["Both"] + sorted(data['Scholarship_str'].unique().tolist())
        debtor_options = ["Both"] + sorted(data['Debtor_str'].unique().tolist())
        actual_status_options = ["All"] + sorted(data['Status'].unique().tolist())
        predicted_status_options = ["All"] + sorted(data['Predicted_Status'].unique().tolist())

        gender_filter = st.selectbox("Gender", options=gender_options)
        scholarship_filter = st.selectbox("Scholarship Holder", options=scholarship_options)
        debtor_filter = st.selectbox("Debtor", options=debtor_options)
        actual_status_filter = st.selectbox("Actual Status", options=actual_status_options)
        predicted_status_filter = st.selectbox("Predicted Status", options=predicted_status_options)

    filtered_data = data.copy()
    if gender_filter != "Both":
        filtered_data = filtered_data[filtered_data['Gender_str'] == gender_filter]
    if scholarship_filter != "Both":
        filtered_data = filtered_data[filtered_data['Scholarship_str'] == scholarship_filter]
    if debtor_filter != "Both":
        filtered_data = filtered_data[filtered_data['Debtor_str'] == debtor_filter]
    if actual_status_filter != "All":
        filtered_data = filtered_data[filtered_data['Status'] == actual_status_filter]
    if predicted_status_filter != "All":
        filtered_data = filtered_data[filtered_data['Predicted_Status'] == predicted_status_filter]

    # Display full table
    display_columns = [
        'Status', 'Predicted_Status',
        'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_approved',
        'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_approved',
        'Tuition_str', 'Age_at_enrollment', 'Scholarship_str', 'Debtor_str', 'Gender_str', 'Recommended Action'
    ]

    st.write("### Recommendation Results:")
    st.dataframe(filtered_data[display_columns])