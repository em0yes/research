import pandas as pd
import glob
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Zone을 알파벳 및 숫자 순서로 정렬하기 위한 함수
def sort_zones(zones):
    return sorted(zones, key=lambda x: (str(x).isdigit(), x))

# 슬라이딩 윈도우를 적용하여 예측을 수행하는 함수
def process_and_predict_per_file(file, best_rf, rfe, le, window_size, step_size, feature_names):
    window_data = pd.read_csv(file)
    
    start_index = 0
    all_predictions = []
    all_true_zones = []
    
    while start_index + window_size <= len(window_data):
        window_slice = window_data.iloc[start_index:start_index + window_size]
        
        # 'TimeStamp'와 'Zone' 컬럼 제외하고 Beacon 데이터만 추출
        beacon_data = window_slice.drop(columns=['TimeStamp', 'Zone'], errors='ignore')
        
        # 0 값을 NaN으로 변환한 후, 각 Beacon 별로 평균을 계산 (NaN은 제외)
        beacon_means = beacon_data.replace(0, np.nan).mean()
        
        # 평균을 계산한 후, feature 이름을 학습 데이터와 일치시킴
        beacon_means = pd.DataFrame([beacon_means], columns=feature_names)
        
        # RFE를 통해 특징 선택
        X_test_rfe = rfe.transform(beacon_means)

        # 예측
        y_pred_encoded = best_rf.predict(X_test_rfe)
        y_pred = le.inverse_transform(y_pred_encoded)

        # 실제 Zone과 예측된 Zone 저장
        true_zone = window_slice['Zone'].iloc[0]  # 첫 번째 행의 실제 Zone 사용
        all_true_zones.append(true_zone)
        all_predictions.append(y_pred[0])

        start_index += step_size

    return all_true_zones, all_predictions

# 예측 결과를 그래프로 그리는 함수
def plot_prediction_results(predictions, actual_zones_windows, filename, output_folder):
    # 예측 및 실제 Zone 값 정렬
    all_zones = sorted(set(predictions).union(set(actual_zones_windows or [])), key=sort_zones)
    zone_labels_sorted = sort_zones(all_zones)

    # 예측 결과를 그래프로 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(predictions)), predictions, label='Predicted Zone', marker='o', linestyle='-', color='b')

    if actual_zones_windows is not None:
        plt.plot(range(len(actual_zones_windows)), actual_zones_windows, label='True Zone', marker='x', linestyle='--', color='r')

    plt.title(f"Zone Prediction Results for {filename}")
    plt.xlabel('Window Index')
    plt.ylabel('Zone')

    # y축에 알파벳 및 숫자 순서대로 정렬된 Zone 적용
    plt.yticks(ticks=range(len(zone_labels_sorted)), labels=zone_labels_sorted)
    plt.legend()
    plt.grid(True)

    # 그래프 저장
    graph_output_path = os.path.join(output_folder, f"{filename.replace('.csv', '')}_prediction_graph.png")
    plt.tight_layout()
    plt.savefig(graph_output_path)
    plt.close()

    print(f"Graph saved as {graph_output_path}")

# 전체 CSV 파일에 대해 처리하고, 결과를 저장하는 함수
def predict_from_folder(window_folder, output_txt_file, output_graph_folder, window_size=5, step_size=5):
    all_csv_files = glob.glob(os.path.join(window_folder, "*.csv"))
    
    # 모델, RFE 선택기, 레이블 인코더 불러오기
    best_rf = joblib.load('_1_RF_AVG_REF(O)_shuffle(O)_model.pkl')
    rfe = joblib.load('_1_RF_AVG_REF(O)_shuffle(O)_selector.pkl')
    le = joblib.load('_1_RF_AVG_REF(O)_shuffle(O)_label_encoder.pkl')

    # 학습 시 사용한 feature 이름 불러오기
    train_df = pd.read_csv("train_data.csv")
    feature_names = train_df.drop(columns=['TimeStamp', 'Zone']).columns

    # 그래프 저장 폴더 확인 및 생성
    if not os.path.exists(output_graph_folder):
        os.makedirs(output_graph_folder)

    # 텍스트 파일을 오픈하여 결과 작성
    with open(output_txt_file, 'w') as f:
        # 각 CSV 파일에 대해 예측 수행
        for file in all_csv_files:
            print(f"Processing file: {file}")
            true_zones, predictions = process_and_predict_per_file(file, best_rf, rfe, le, window_size, step_size, feature_names)
            
            # 정확도 계산
            accuracy = accuracy_score(true_zones, predictions)
            classification_rep = classification_report(true_zones, predictions, zero_division=1)  # zero_division=1 추가
            
            # 결과 저장
            f.write(f"File: {os.path.basename(file)}\n")
            f.write(f"Accuracy: {accuracy:.2f}\n")
            f.write("Classification Report:\n")
            f.write(f"{classification_rep}\n")
            f.write("="*40 + "\n")

            print(f"File: {file}, Accuracy: {accuracy:.2f}")
            
            # 그래프 저장
            plot_prediction_results(predictions, true_zones, os.path.basename(file), output_graph_folder)

    print(f"Accuracy results saved to: {output_txt_file}")

if __name__ == "__main__":
    test_window_folder = './_test_dataset_transformed'  # 테스트 슬라이딩 윈도우 CSV 파일들이 있는 폴더
    output_txt_file = './test_accuracy_results.txt'
    output_graph_folder = './_1_RF_AVG_REF(O)_shuffle(O)_prediction_results'
    
    # 슬라이딩 윈도우와 스텝 사이즈를 설정하여 테스트 데이터 예측 수행
    window_size = 5  # 원하는 윈도우 크기로 설정
    step_size = 1   # 원하는 스텝 크기로 설정
    
    predict_from_folder(test_window_folder, output_txt_file, output_graph_folder, window_size=window_size, step_size=step_size)
