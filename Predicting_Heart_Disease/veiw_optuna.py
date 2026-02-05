from optuna_dashboard import run_server
import os

db_path = "sqlite:///heart_disease_optuna.db"
print(f"서버를 실행합니다: {db_path}")
run_server(db_path)