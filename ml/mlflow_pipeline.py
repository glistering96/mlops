from datetime import datetime
import mlflow
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MLflow 설정
EXPERIMENT_NAME = "autogluon_iris_classification"
mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment(EXPERIMENT_NAME)

def prepare_data():
    # Iris 데이터셋 로드 및 전처리
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # 데이터프레임 합치기
    df = pd.concat([X, y], axis=1)
    
    return train_test_split(df, test_size=0.2, random_state=42)

def train_autogluon(train_data, **kwargs):
    # AutoGluon 모델 학습
    predictor = TabularPredictor(
        label='target',
        eval_metric='accuracy',
        path='autogluon_model'
    )
    
    predictor.fit(
        train_data=train_data,
        **kwargs
    )
    
    return predictor

def evaluate_model(predictor, test_data):
    # 예측 및 평가
    y_test = test_data['target']
    y_pred = predictor.predict(test_data.drop('target', axis=1))
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics

def log_leaderboard(predictor):
    # AutoGluon 리더보드를 CSV로 저장하고 MLflow에 기록
    leaderboard = predictor.leaderboard()
    leaderboard.to_csv('leaderboard.csv')
    mlflow.log_artifact('leaderboard.csv')

def main():
    # 데이터 준비
    train_data, test_data = prepare_data()
    
    # AutoGluon 학습 설정
    ag_params = {
        "dynamic_stacking": False,  # takes too much time
        "included_model_types": ['GBM', 'CAT', 'RF', 'XT'],
        "presets": "medium_quality",
        "fit_strategy": 'parallel',
        "num_cpus": 4,
    }
    
    # MLflow 실험 시작
    with mlflow.start_run(run_name=f"autogluon_iris_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # 파라미터 로깅
        mlflow.log_params(ag_params)
        mlflow.log_param("train_size", len(train_data))
        mlflow.log_param("test_size", len(test_data))
        
        # AutoGluon 모델 학습
        predictor = train_autogluon(
            train_data,
            test_data,
            **ag_params
        )
        
        # 모델 평가
        metrics = evaluate_model(predictor, test_data)
        mlflow.log_metrics(metrics)
        
        # 리더보드 로깅
        log_leaderboard(predictor)
        
        # 모델 아티팩트 저장
        model_path = "autogluon_model"
        mlflow.log_artifact(model_path)
        
        # 특성 중요도 저장 (가능한 경우)
        try:
            importance = predictor.feature_importance(test_data)
            importance.to_csv('feature_importance.csv')
            mlflow.log_artifact('feature_importance.csv')
        except:
            print("특성 중요도를 계산할 수 없습니다.")

if __name__ == "__main__":
    main()