import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import os
import psutil
import socket
import time
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, auth
import sqlalchemy as sa
from google.cloud import monitoring_v3, logging as cloud_logging
from google.cloud.logging_v2.resource import Resource

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Cloud Logging
cloud_logging_client = cloud_logging.Client()
cloud_logging_handler = cloud_logging.handlers.CloudLoggingHandler(cloud_logging_client)
cloud_logger = logging.getLogger('cloudLogger')
cloud_logger.setLevel(logging.INFO)
cloud_logger.addHandler(cloud_logging_handler)

# Define constants
PROJECT_NAME = "LLM Detect Project"
OWNER = "Gowtham Ram G23AI2029"


# Initialize Firebase
@st.cache_resource
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase-adminsdk.json")
        firebase_admin.initialize_app(cred)
    return firebase_admin.get_app()


firebase_app = initialize_firebase()

# Database setup
db_user = os.environ.get('DB_USER')
db_pass = os.environ.get('DB_PASS')
db_name = os.environ.get('DB_NAME')
db_socket_dir = os.environ.get("DB_SOCKET_DIR", "/cloudsql")
instance_connection_name = os.environ.get("INSTANCE_CONNECTION_NAME")

engine = create_engine(
    f"mysql+pymysql://{db_user}:{db_pass}@/{db_name}?unix_socket={db_socket_dir}/{instance_connection_name}"
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True)
    firebase_uid = Column(String(128), unique=True)
    essay_results = relationship("EssayResult", back_populates="user")


class EssayResult(Base):
    __tablename__ = "essay_results"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    essay = Column(String(1000))
    result = Column(String(20))
    probability = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="essay_results")


def init_db():
    inspector = inspect(engine)
    if not inspector.has_table("users"):
        Base.metadata.tables["users"].create(bind=engine)
    if not inspector.has_table("essay_results"):
        Base.metadata.tables["essay_results"].create(bind=engine)
    else:
        columns = inspector.get_columns("essay_results")
        if "user_id" not in [column['name'] for column in columns]:
            with engine.begin() as connection:
                connection.execute(sa.text(
                    "ALTER TABLE essay_results ADD COLUMN user_id INTEGER"
                ))
                connection.execute(sa.text(
                    "ALTER TABLE essay_results ADD CONSTRAINT fk_user_id FOREIGN KEY (user_id) REFERENCES users(id)"
                ))


def create_default_user():
    default_email = "admin@example.com"
    default_password = "admin123"
    db = SessionLocal()
    existing_user = db.query(User).filter(User.email == default_email).first()
    if not existing_user:
        try:
            firebase_user = auth.create_user(email=default_email, password=default_password)
            db_user = User(email=default_email, firebase_uid=firebase_user.uid)
            db.add(db_user)
            db.commit()
            cloud_logger.info(f"Default user created: {default_email}")
        except Exception as e:
            cloud_logger.error(f"Error creating default user: {str(e)}")
    db.close()


def preprocess(essay: str) -> str:
    return essay.lower().replace('\n', '').replace('\t', '').replace(u'\xa0', u' ')


@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("import of model from gemini api from huggingface")
    model = AutoModelForSequenceClassification.from_pretrained("#import of model from google.cloud.vision from huggingface")
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()


def get_cloud_info():
    try:
        return {
            "Hostname": socket.gethostname(),
            "IP Address": socket.gethostbyname(socket.gethostname()),
            "CPU Usage": f"{psutil.cpu_percent(interval=1):.1f}%",
            "Memory Usage": f"{psutil.virtual_memory().percent:.1f}%",
            "Disk Usage": f"{psutil.disk_usage('/').percent:.1f}%",
            "Cloud Run Service": os.environ.get('K_SERVICE', 'N/A'),
            "Cloud Run Revision": os.environ.get('K_REVISION', 'N/A'),
            "Cloud Run Configuration": os.environ.get('K_CONFIGURATION', 'N/A'),
            "Service Account": os.environ.get('K_SERVICE_ACCOUNT', 'N/A'),
        }
    except Exception as e:
        cloud_logger.error(f"Error getting cloud info: {str(e)}")
        return {"Error": "Unable to retrieve cloud information"}


def create_custom_metric(project_id, metric_type, value):
    try:
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{project_id}"
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/{metric_type}"
        series.resource.type = "global"

        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10 ** 9)
        interval = monitoring_v3.TimeInterval(
            {
                "end_time": {"seconds": seconds, "nanos": nanos},
            }
        )
        point = monitoring_v3.Point({"interval": interval, "value": {"double_value": value}})
        series.points = [point]

        client.create_time_series(name=project_name, time_series=[series])
        return True
    except Exception as e:
        cloud_logger.error(f"Error creating custom metric {metric_type}: {str(e)}")
        cloud_logger.error(f"Project ID: {project_id}")
        cloud_logger.error(f"Metric Type: {metric_type}")
        cloud_logger.error(f"Value: {value}")
        return False


def sign_up():
    with st.form("Sign Up"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign Up")
        if submit:
            try:
                user = auth.create_user(email=email, password=password)
                db = SessionLocal()
                db_user = User(email=email, firebase_uid=user.uid)
                db.add(db_user)
                db.commit()
                cloud_logger.info(f"New user signed up: {email}")
                st.success("Account created successfully!")
                return user
            except Exception as e:
                cloud_logger.error(f"Error during sign up: {str(e)}")
                st.error(f"Error: {str(e)}")
    return None


def sign_in():
    with st.form("Sign In"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign In")
        if submit:
            try:
                user = auth.get_user_by_email(email)
                cloud_logger.info(f"User signed in: {email}")
                st.success("Signed in successfully!")
                return user
            except Exception as e:
                cloud_logger.error(f"Error during sign in: {str(e)}")
                st.error(f"Error: {str(e)}")
    return None


def main():
    init_db()
    create_default_user()
    st.title(PROJECT_NAME)
    st.write(f"Developed by {OWNER}")

    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        st.error("GOOGLE_CLOUD_PROJECT environment variable is not set.")
        cloud_logger.error("GOOGLE_CLOUD_PROJECT environment variable is not set.")
    else:
        st.sidebar.text(f"Project ID: {project_id}")

    # Initialize metric_logging_success
    metric_logging_success = False

    if 'user' not in st.session_state:
        st.session_state.user = None

    if not st.session_state.user:
        choice = st.radio("Choose an option", ["Sign In", "Sign Up"])
        if choice == "Sign Up":
            user = sign_up()
        else:
            user = sign_in()
        if user:
            st.session_state.user = user

    if st.session_state.user:
        st.write(f"Welcome, {st.session_state.user.email}")

        st.sidebar.title("Cloud Environment Info")
        cloud_info = get_cloud_info()
        for key, value in cloud_info.items():
            st.sidebar.text(f"{key}: {value}")

        st.subheader("LLM Detection")
        with st.form(key='llm_detect_form'):
            essay_text = st.text_area("Enter the Essay:", height=250)
            submit_button = st.form_submit_button('Analyze Essay')

        if submit_button and essay_text:
            start_time = time.time()
            processed_essay = preprocess(essay_text)
            inputs = tokenizer(processed_essay, padding='max_length', truncation=True, max_length=512,
                               return_tensors='pt')
            with torch.no_grad():
                logits = model(**inputs).logits
                probability = nn.functional.sigmoid(logits).item()

            result_text = "LLM Written" if probability > 0.71 else "Not LLM Written"
            result_color = "red" if probability > 0.71 else "green"
            st.markdown(f"<h2 style='text-align: center; color: {result_color};'>{result_text}</h2>",
                        unsafe_allow_html=True)
            st.write(f"Probability: {probability:.2f}")
            st.write("Note: This is simply a prediction and can be incorrect.")

            db = SessionLocal()
            user = db.query(User).filter(User.firebase_uid == st.session_state.user.uid).first()
            new_result = EssayResult(user_id=user.id, essay=essay_text[:1000], result=result_text,
                                     probability=probability)
            db.add(new_result)
            db.commit()
            db.close()

            processing_time = time.time() - start_time
            st.write(f"Processing Time: {processing_time:.2f} seconds")

            # Log custom metrics
            try:
                metric_logging_success = create_custom_metric(project_id, "essay_length", len(essay_text))
                if metric_logging_success:
                    metric_logging_success = create_custom_metric(project_id, "processing_time", processing_time)
            except Exception as e:
                cloud_logger.error(f"Error logging metrics: {str(e)}")
                metric_logging_success = False

            if not metric_logging_success:
                st.warning("Custom metric logging failed. This does not affect the analysis result.")

            # Log essay analysis details
            log_message = (
                f"Essay analyzed. "
                f"Result: {result_text}, "
                f"Probability: {probability:.4f}, "
                f"Processing Time: {processing_time:.2f}s, "
                f"Essay Length: {len(essay_text)}, "
                f"Metric Logging Success: {metric_logging_success}"
            )
            cloud_logger.info(log_message)

        st.subheader("Your Past Results")
        db = SessionLocal()
        user = db.query(User).filter(User.firebase_uid == st.session_state.user.uid).first()
        past_results = db.query(EssayResult).filter(EssayResult.user_id == user.id).order_by(
            EssayResult.timestamp.desc()).limit(5).all()
        db.close()

        for result in past_results:
            st.write(f"Date: {result.timestamp}")
            st.write(f"Result: {result.result}")
            st.write(f"Probability: {result.probability:.2f}")
            st.write("Essay:")
            st.text_area("", result.essay, height=100, disabled=True)
            st.write("---")

        st.sidebar.title("Cloud Metrics")
        st.sidebar.text(f"Max Memory: {psutil.virtual_memory().total / (1024.0 ** 3):.2f} GB")
        st.sidebar.text(f"Available Memory: {psutil.virtual_memory().available / (1024.0 ** 3):.2f} GB")
        st.sidebar.text(f"CPU Cores: {psutil.cpu_count()}")
        st.sidebar.text(f"Custom Metric Logging: {'Enabled' if metric_logging_success else 'Failed'}")

        st.sidebar.title("Cloud Features Used")
        st.sidebar.text("- Google Cloud Run")
        st.sidebar.text("- Cloud SQL")
        st.sidebar.text("- Cloud Monitoring")
        st.sidebar.text("- Cloud Logging")
        st.sidebar.text("- Firebase Authentication")


if __name__ == "__main__":
    main()