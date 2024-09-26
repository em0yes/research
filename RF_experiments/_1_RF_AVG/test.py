import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import re
import joblib  # joblib을 사용하여 모델을 저장하고 불러오기 위해 필요
import numpy as np

class ZoneClassifierWithSlidingWindow:
    def __init__(self, window_size_rows=5, step_size_rows=1):
        self.window_size = window_size_rows
        self.step_size_rows = step_size_rows
        self.model = None  # 모델 로드 후 할당
        self.label_encoder = None  # 레이블 인코더 로드 후 할당
        self.feature_names = None  # 피처 이름 저장

    def load_model(self, model_file, encoder_file):
        """
        저장된 모델과 레이블 인코더를 불러옵니다.
        """
        # 모델 로드
        self.model = joblib.load(model_file)
        print(f"모델이 {model_file}에서 로드되었습니다.")

        # 레이블 인코더 로드
        self.label_encoder = joblib.load(encoder_file)
        print(f"레이블 인코더가 {encoder_file}에서 로드되었습니다.")

    def preprocess_data(self, data):
        """
        새로운 데이터에 대해 슬라이딩 윈도우 방식으로 전처리.
        유효한 슬라이딩 윈도우를 추출하고 모델 예측에 사용할 수 있도록 준비합니다.
        """
        # 'TimeStamp'와 'Zone' 컬럼 제외하고 Beacon 데이터만 추출
        beacon_columns = [col for col in data.columns if col not in ['TimeStamp', 'Zone']]
        self.feature_names = beacon_columns  # 피처 이름 저장

        # 슬라이딩 윈도우 처리
        start_index = 0
        grouped_data = []

        while start_index + self.window_size <= len(data):
            window_data = data.iloc[start_index:start_index + self.window_size]

            # 0 값을 NaN으로 변환한 후, 각 Beacon 별로 평균을 계산 (NaN은 제외)
            window_data_no_zero = window_data[beacon_columns].replace(0, np.nan)
            beacon_means = window_data_no_zero.mean()

            grouped_data.append(beacon_means.tolist())

            start_index += self.step_size_rows

        return pd.DataFrame(grouped_data, columns=self.feature_names)

    def predict(self, data):
        """
        새로운 데이터에 대해 예측을 수행합니다.
        """
        # 데이터 전처리 (슬라이딩 윈도우 적용)
        X_test = self.preprocess_data(data)

        # 예측 수행 (숫자 레이블로 예측)
        predictions_encoded = self.model.predict(X_test)

        # 예측된 숫자 레이블을 다시 문자열로 변환
        predictions = self.label_encoder.inverse_transform(predictions_encoded)

        return predictions

def sort_zones(zones):
    """
    Zone 리스트를 알파벳과 숫자 순서대로 정렬하는 함수.
    Zone 값은 'A_1', 'B_2'와 같은 형식이어야 합니다.
    """
    def zone_key(zone):
        match = re.match(r"([A-Za-z]+)_(\d+)", zone)
        if match:
            return (match.group(1), int(match.group(2)))
        return zone  # 매치가 안 될 경우 그대로 반환

    return sorted(zones, key=zone_key)

# 새로운 데이터에 대해 예측하는 메인 함수
def main(test_folder, model_file, encoder_file):
    # 슬라이딩 윈도우 기반 분류기 인스턴스 생성
    classifier = ZoneClassifierWithSlidingWindow(window_size_rows=5, step_size_rows=1)

    # 저장된 모델과 레이블 인코더 로드
    classifier.load_model(model_file, encoder_file)

    # 출력 폴더 설정
    output_folder = './RF_10_prediction_results'
    os.makedirs(output_folder, exist_ok=True)
    accuracy_output_file = os.path.join(output_folder, "prediction_accuracies.txt")

    with open(accuracy_output_file, "w") as accuracy_file:
        for filename in os.listdir(test_folder):
            if filename.endswith('.csv'):
                test_file = os.path.join(test_folder, filename)

                # 테스트 데이터 로드
                test_data = pd.read_csv(test_file)

                # 슬라이딩 윈도우 처리
                X_test = classifier.preprocess_data(test_data)

                # 실제 Zone 값 처리
                if 'Zone' in test_data.columns:
                    actual_zones = test_data['Zone']
                    actual_zones_windows = []
                    start_index = 0
                    while start_index + classifier.window_size <= len(actual_zones):
                        window_zone = actual_zones.iloc[start_index:start_index + classifier.window_size]
                        # 첫 번째 행의 Zone 값을 실제 값으로 사용
                        actual_zones_windows.append(window_zone.iloc[0])
                        start_index += classifier.step_size_rows
                else:
                    actual_zones_windows = None

                # 예측 수행
                predictions = classifier.predict(test_data)

                # 결과 출력 및 텍스트 파일로 저장
                if actual_zones_windows is not None and len(predictions) == len(actual_zones_windows):
                    accuracy = accuracy_score(actual_zones_windows, predictions)
                    classification_rep = classification_report(actual_zones_windows, predictions, zero_division=1)
                    accuracy_file.write(f"File: {filename}\n")
                    accuracy_file.write(f"Accuracy: {accuracy:.2f}\n")
                    accuracy_file.write("Classification Report:\n")
                    accuracy_file.write(f"{classification_rep}\n")
                    accuracy_file.write("="*40 + "\n")
                    print(f"File: {filename}, Accuracy: {accuracy:.2f}")
                else:
                    print(f"Skipping accuracy calculation for {filename} due to mismatched lengths.")
                    accuracy_file.write(f"File: {filename}\n")
                    accuracy_file.write("Accuracy calculation skipped due to mismatched lengths.\n")
                    accuracy_file.write("="*40 + "\n")

                # 예측 및 실제 Zone 값 정렬
                all_zones = sorted(set(predictions).union(set(actual_zones_windows or [])), key=sort_zones)
                zone_labels_sorted = sort_zones(all_zones)

                # 예측 결과 그래프로 저장
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

                graph_output_path = os.path.join(output_folder, f"{filename.replace('.csv', '')}_prediction_graph.png")
                plt.tight_layout()
                plt.savefig(graph_output_path)
                plt.close()
                print(f"Graph saved as {graph_output_path}")

    print(f"Accuracy results saved to: {accuracy_output_file}")

# 사용 예시:
if __name__ == "__main__":
    test_folder = './_test_dataset_transformed'  # 테스트 데이터가 들어있는 폴더 경로
    model_file = './RF_10_model.pkl'  # 학습된 모델 파일 경로
    encoder_file = './RF_10_label_encoder.pkl'  # 레이블 인코더 파일 경로

    # 파이프라인 실행
    main(test_folder, model_file, encoder_file)
