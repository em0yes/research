import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils import shuffle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# 슬라이딩 윈도우 및 데이터셋 처리 함수들
def save_sliding_windows(data_file, output_folder, file_id, window_size_rows=5, step_size_rows=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data = pd.read_csv(data_file)
    data = data.sort_values(by='TimeStamp')
    start_index = 0
    window_counter = 0
    while start_index + window_size_rows <= len(data):
        window_data = data.iloc[start_index:start_index + window_size_rows]
        if not window_data.empty and len(window_data) == window_size_rows:
            window_data_filled = window_data.fillna(0)
            window_data_filled = window_data_filled[['TimeStamp'] + [col for col in window_data_filled.columns if col != 'TimeStamp']]
            window_file = os.path.join(output_folder, f"window_{file_id}_{window_counter}.csv")
            window_data_filled.to_csv(window_file, index=False)
            window_counter += 1
        start_index += step_size_rows

def process_folder(input_folder, output_folder, window_size_rows=5, step_size_rows=1):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    for idx, file in enumerate(csv_files):
        print(f"Processing file: {file}")
        save_sliding_windows(file, output_folder, file_id=idx, window_size_rows=window_size_rows, step_size_rows=step_size_rows)

def split_and_save_files(input_folder, train_folder, val_folder, split_ratio=0.8):
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    all_csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    train_files, val_files = train_test_split(all_csv_files, train_size=split_ratio, random_state=42)
    for file in train_files:
        file_name = os.path.basename(file)
        os.rename(file, os.path.join(train_folder, file_name))
    for file in val_files:
        file_name = os.path.basename(file)
        os.rename(file, os.path.join(val_folder, file_name))

def calculate_mean_per_window_and_save_train(window_folder, output_file):
    all_csv_files = glob.glob(os.path.join(window_folder, "*.csv"))
    mean_list = []
    for file in all_csv_files:
        window_data = pd.read_csv(file)
        beacon_data = window_data.drop(columns=['TimeStamp', 'Zone'])
        beacon_means = beacon_data.replace(0, np.nan).mean()
        beacon_means = beacon_means.fillna(0).round(0).astype(int)
        beacon_means['Zone'] = window_data['Zone'].iloc[0]
        beacon_means['TimeStamp'] = window_data['TimeStamp'].iloc[0]
        mean_list.append(beacon_means)
    mean_df = pd.DataFrame(mean_list)
    shuffled_mean_df = shuffle(mean_df)
    columns_order = ['TimeStamp'] + [col for col in shuffled_mean_df.columns if col != 'TimeStamp']
    shuffled_mean_df = shuffled_mean_df[columns_order]
    shuffled_mean_df.to_csv(output_file, index=False)
    print(f"Shuffled mean values saved to: {output_file}")

def calculate_mean_per_window_and_save_val(window_folder, output_file):
    all_csv_files = glob.glob(os.path.join(window_folder, "*.csv"))
    mean_list = []
    for file in all_csv_files:
        window_data = pd.read_csv(file)
        beacon_data = window_data.drop(columns=['TimeStamp', 'Zone'])
        beacon_means = beacon_data.replace(0, np.nan).mean()
        beacon_means = beacon_means.fillna(0).round(0).astype(int)
        beacon_means['Zone'] = window_data['Zone'].iloc[0]
        beacon_means['TimeStamp'] = window_data['TimeStamp'].iloc[0]
        mean_list.append(beacon_means)
    mean_df = pd.DataFrame(mean_list)
    columns_order = ['TimeStamp'] + [col for col in mean_df.columns if col != 'TimeStamp']
    mean_df = mean_df[columns_order]
    mean_df.to_csv(output_file, index=False)
    print(f"Mean values saved to: {output_file}")

# Random Forest 모델 학습 및 최적화
def train_random_forest():
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

    # 학습된 모델과 레이블 인코더 저장
    joblib.dump(best_rf, '_1_RF_AVG_shuffle(O)_model.pkl')
    joblib.dump(le, '_1_RF_AVG_shuffle(O)_label_encoder.pkl')

    print("Optimized model and label encoder saved.")

if __name__ == "__main__":
    # 1. 슬라이딩 윈도우 처리 및 저장
    input_data_folder = './_dataset_transformed'
    output_window_folder = './processed_sliding_windows'
    process_folder(input_data_folder, output_window_folder, window_size_rows=5, step_size_rows=1)

    # 2. 슬라이딩 윈도우 파일을 train과 val로 나누어 저장
    train_folder = './train_windows'
    val_folder = './val_windows'
    split_and_save_files(output_window_folder, train_folder, val_folder)

    # 3. train 데이터 윈도우의 평균을 구하고 셔플 후 저장
    train_mean_file = './train_data.csv'
    calculate_mean_per_window_and_save_train(train_folder, train_mean_file)

    # 4. val 데이터 윈도우의 평균을 구하고 셔플 없이 저장
    val_mean_file = './val_data.csv'
    calculate_mean_per_window_and_save_val(val_folder, val_mean_file)

    # 5. Random Forest 학습 및 모델 저장
    train_random_forest()
