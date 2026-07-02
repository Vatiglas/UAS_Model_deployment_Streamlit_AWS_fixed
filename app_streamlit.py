import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

CLASS_COLORS = {
    "Good":     "#2ecc71",
    "Standard": "#3498db",
    "Poor":     "#e74c3c",
}
CLASS_DESCRIPTIONS = {
    "Good":     "Nasabah memiliki riwayat kredit yang sangat baik. Risiko rendah.",
    "Standard": "Nasabah memiliki riwayat kredit yang cukup baik. Risiko moderat.",
    "Poor":     "Nasabah memiliki riwayat kredit yang buruk. Risiko tinggi.",
}

OCCUPATION_OPTIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer",
    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
    "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
]
PAYMENT_BEHAVIOUR_OPTIONS = [
    "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
]


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(input_data: dict) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [input_data]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.set_page_config(page_title="Credit Score Classifier", page_icon="", layout="wide")

st.title("Credit Score Classifier")
st.write(
    f"Prediksi performa kredit nasabah via AWS SageMaker endpoint "
    f"(`{ENDPOINT_NAME}`, region `{REGION}`)."
)
st.divider()

# ── User Inputs ─────────────────────────────────────────────────────────────
st.subheader("Profil Nasabah")
col1, col2, col3, col4 = st.columns(4)
with col1:
    age = st.number_input("Usia", min_value=10, max_value=100, value=35)
with col2:
    occupation = st.selectbox("Pekerjaan", OCCUPATION_OPTIONS, index=4)
with col3:
    credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"], index=1)
with col4:
    payment_min = st.selectbox("Hanya Bayar Minimum?", ["No", "Yes"], index=0)

st.subheader("Pendapatan & Tabungan")
col5, col6, col7, col8 = st.columns(4)
with col5:
    annual_income = st.number_input("Pendapatan Tahunan (USD)", min_value=0.0, value=50000.0, step=1000.0)
with col6:
    monthly_salary = st.number_input("Gaji Bersih Bulanan (USD)", min_value=0.0, value=4000.0, step=100.0)
with col7:
    monthly_balance = st.number_input("Saldo Bulanan (USD)", value=500.0, step=50.0)
with col8:
    amount_invested = st.number_input("Investasi Bulanan (USD)", min_value=0.0, value=100.0, step=10.0)

st.subheader("Profil Kredit")
col9, col10, col11, col12 = st.columns(4)
with col9:
    num_bank_accounts = st.number_input("Jumlah Rekening Bank", min_value=0, value=3)
with col10:
    num_credit_card = st.number_input("Jumlah Kartu Kredit", min_value=0, value=3)
with col11:
    interest_rate = st.number_input("Suku Bunga (%)", min_value=0.0, value=15.0, step=0.5)
with col12:
    num_loan = st.number_input("Jumlah Pinjaman Aktif", min_value=0, value=2)

col13, col14, col15, col16 = st.columns(4)
with col13:
    outstanding_debt = st.number_input("Total Utang Outstanding (USD)", min_value=0.0, value=1000.0, step=100.0)
with col14:
    credit_util = st.number_input("Rasio Penggunaan Kredit (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
with col15:
    total_emi = st.number_input("Total Cicilan/Bulan (USD)", min_value=0.0, value=200.0, step=10.0)
with col16:
    credit_history = st.number_input("Lama Riwayat Kredit (bulan)", min_value=0, value=60)

st.subheader("Riwayat Pembayaran")
col17, col18, col19, col20 = st.columns(4)
with col17:
    delay_due_date = st.number_input("Rata-rata Keterlambatan (hari)", min_value=0, value=10)
with col18:
    num_delayed = st.number_input("Jumlah Pembayaran Terlambat", min_value=0, value=5)
with col19:
    changed_limit = st.number_input("Perubahan Limit Kredit (%)", value=5.0, step=0.5)
with col20:
    num_inquiries = st.number_input("Jumlah Inquiry Kredit", min_value=0, value=3)

payment_behaviour = st.selectbox("Perilaku Pembayaran", PAYMENT_BEHAVIOUR_OPTIONS, index=4)

if st.button("🔍 Predict", type="primary"):
    input_data = {
        "Age": age,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_loan,
        "Delay_from_due_date": delay_due_date,
        "Num_of_Delayed_Payment": num_delayed,
        "Changed_Credit_Limit": changed_limit,
        "Num_Credit_Inquiries": num_inquiries,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_util,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": amount_invested,
        "Monthly_Balance": monthly_balance,
        "Credit_History_Months": credit_history,
        "Credit_Mix": credit_mix,
        "Occupation": occupation,
        "Payment_Behaviour": payment_behaviour,
        "Payment_of_Min_Amount": payment_min,
    }

    try:
        result = invoke_endpoint(input_data)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["labels"][0]
        proba_dict = result["probabilities"][0] 
        color = CLASS_COLORS[label]

        st.success(f"Predicted Credit Score: **{label}**")
        st.markdown(
            f"<p style='color:{color}; font-size:1.1rem;'>{CLASS_DESCRIPTIONS[label]}</p>",
            unsafe_allow_html=True,
        )

        st.write("Class probabilities (%):")
        st.bar_chart(proba_dict)
