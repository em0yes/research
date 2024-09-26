import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Random Forest 모델 학습 및 최적화
def train_random_forest(train_file, val_file, model_output_file, label_encoder_output_file):
    # 데이터 불러오기
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # 특징 변수(X)와 목표 변수(y) 설정
    X_train = train_df.drop(columns=['TimeStamp', 'Zone'])
    y_train = train_df['Zone']

    X_val = val_df.drop(columns=['TimeStamp', 'Zone'])
    y_val = val_df['Zone']

    # Label Encoding을 사용해 Zone 열을 숫자로 변환
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)

    # Random Forest 하이퍼파라미터 범위 설정
    param_dist = {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    # Random Forest 모델 생성
    rf = RandomForestClassifier(random_state=42)

    # RandomizedSearchCV로 랜덤 포레스트 최적화
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train_encoded)

    # 최적의 하이퍼파라미터 출력
    print(f"Best Parameters: {rf_random.best_params_}")

    # 최적화된 Random Forest 모델 학습
    best_rf = rf_random.best_estimator_
    best_rf.fit(X_train, y_train_encoded)

    # 검증 데이터로 예측
    y_pred_encoded = best_rf.predict(X_val)

    # 예측된 결과를 다시 문자열로 변환
    y_pred = le.inverse_transform(y_pred_encoded)

    # 성능 평가
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    # 학습된 모델과 레이블 인코더 저장 (pickle 사용)
    with open(model_output_file, 'wb') as f:
        pickle.dump(best_rf, f)
    
    with open(label_encoder_output_file, 'wb') as f:
        pickle.dump(le, f)

    print(f"Optimized model saved to {model_output_file}")
    print(f"Label encoder saved to {label_encoder_output_file}")

if __name__ == "__main__":
    train_file = './train_data.csv'
    val_file = './val_data.csv'
    model_output_file = 'RF_model.pkl'
    label_encoder_output_file = 'RF_label_encoder.pkl'

    # 모델 학습 및 저장
    train_random_forest(train_file, val_file, model_output_file, label_encoder_output_file)
