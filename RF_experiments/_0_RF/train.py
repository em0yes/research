import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# 데이터 불러오기
train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")

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
    'n_estimators': [100, 200, 500, 1000],        # 트리의 개수
    'max_depth': [10, 20, 30, None],              # 트리의 최대 깊이
    'min_samples_split': [2, 5, 10],              # 노드 분할을 위한 최소 샘플 수
    'min_samples_leaf': [1, 2, 4],                # 리프 노드에 있어야 하는 최소 샘플 수
    'max_features': ['sqrt', 'log2'],             # 각 분할에서 고려할 최대 특징 수
    'bootstrap': [True, False]                    # 부트스트랩 샘플링 여부
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

# 학습된 모델과 레이블 인코더 저장
joblib.dump(best_rf, '_0_RF_shuffle(O)_model.pkl')
joblib.dump(le, '_0_RF_shuffle(O)_label_encoder.pkl')

print("Optimized model and label encoder saved.")
