import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

class ZoneClassifierWithSlidingWindow:
    def __init__(self, window_size_rows=5, step_size_rows=5):
        # DecisionTreeClassifier 초기화 (기본 설정 사용)
        self.model = DecisionTreeClassifier(random_state=42)
        self.window_size = window_size_rows  # 슬라이딩 윈도우 크기 (데이터 개수 기준)
        self.step_size = step_size_rows  # 슬라이딩 윈도우 이동 간격 (데이터 개수 기준)
        self.label_encoder = LabelEncoder()  # LabelEncoder 초기화

    def load_data(self, train_file, test_file):
        """
        학습 및 검증 데이터를 CSV 파일에서 불러옵니다.
        """
        self.train_data = pd.read_csv(train_file)
        self.val_data = pd.read_csv(test_file)
        
        # Zone 열을 Label Encoding 처리
        self.train_data['Zone'] = self.label_encoder.fit_transform(self.train_data['Zone'])
        self.val_data['Zone'] = self.label_encoder.transform(self.val_data['Zone'])

    def preprocess_data(self, data):
        """
        데이터를 슬라이딩 윈도우 방식으로 데이터를 묶어서 그룹화합니다.
        윈도우 내에 모든 Zone이 동일한 경우에만 유효한 윈도우로 간주하고,
        이를 리스트로 반환합니다.
        """
        data = data.sort_values(by='TimeStamp')

        grouped_data = []
        grouped_labels = []
        start_index = 0  # 첫 시작 index

        # 슬라이딩 윈도우 방식으로 그룹화 (step_size_rows 만큼 이동)
        while start_index + self.window_size <= len(data):
            # 윈도우 크기만큼 데이터 묶기
            window_data = data.iloc[start_index:start_index + self.window_size]

            # 윈도우 내 Zone 값이 모두 동일한지 확인
            if not window_data.empty and window_data['Zone'].nunique() == 1:
                # 유효한 윈도우만 데이터로 처리
                window_data_no_zero = window_data.replace(0, pd.NA)
                grouped_data.append(window_data_no_zero.drop(columns=['TimeStamp', 'Zone']).mean(skipna=True).tolist())
                grouped_labels.append(window_data['Zone'].iloc[-1])

            # 슬라이딩 윈도우 이동
            start_index += self.step_size

        return pd.DataFrame(grouped_data), grouped_labels

    def train_model(self):
        """
        학습 데이터를 통해 Decision Tree 모델을 학습합니다.
        """
        # 슬라이딩 윈도우로 학습 데이터를 전처리
        self.X_train, self.y_train = self.preprocess_data(self.train_data)

        # 모델 학습
        self.model.fit(self.X_train, self.y_train)

        print("모델 학습 완료")

    def validate_model(self, report_file):
        """
        검증 데이터를 통해 모델을 검증하고 성능을 출력하고, 결과를 파일로 저장합니다.
        """
        # 슬라이딩 윈도우로 검증 데이터를 전처리
        self.X_val, self.y_val = self.preprocess_data(self.val_data)

        # 검증 데이터를 이용하여 Zone 예측
        y_pred_encoded = self.model.predict(self.X_val)

        # 예측된 결과를 다시 원래 문자열로 변환
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_val_original = self.label_encoder.inverse_transform(self.y_val)

        # 정확도 계산
        accuracy = accuracy_score(y_val_original, y_pred)
        classification_report_str = classification_report(y_val_original, y_pred)

        # 콘솔에 결과 출력
        print(f"검증 정확도: {accuracy:.2f}")
        print("분류 보고서:")
        print(classification_report_str)

        # 결과를 파일에 저장
        with open(report_file, 'w') as f:
            f.write(f"검증 정확도: {accuracy:.2f}\n")
            f.write("분류 보고서:\n")
            f.write(classification_report_str)

        print(f"검증 결과와 분류 보고서가 {report_file}에 저장되었습니다.")

    def save_model(self, model_file):
        """
        학습된 모델과 Encoder를 파일로 저장합니다.
        """
        # 모델 저장
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # LabelEncoder 저장 (모델 파일 이름을 기반으로 인코더 파일 이름 설정)
        encoder_file = model_file.replace('_model.pkl', '_encoder.pkl')
        with open(encoder_file, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"모델이 {model_file}에, Encoder가 {encoder_file}에 저장되었습니다.")

# 전체 파이프라인을 실행하는 메인 함수
def main(train_file, test_file, model_name, report_file):
    # 슬라이딩 윈도우 기반 분류기 인스턴스 생성
    classifier = ZoneClassifierWithSlidingWindow(window_size_rows=5, step_size_rows=1)  # 5개씩 묶고 1개씩 이동

    # 1단계: 데이터를 불러오기
    classifier.load_data(train_file, test_file)

    # 2단계: 학습 모델 학습
    classifier.train_model()

    # 3단계: 모델 검증 및 성능 확인, 결과 파일로 저장
    classifier.validate_model(report_file)

    # 4단계: 학습된 모델 저장 (모델 파일과 인코더 파일을 각각 저장)
    model_file = f'./models/{model_name}_model.pkl'
    classifier.save_model(model_file)

# 사용 예시:
train_file = './data/train_data.csv'  # 학습 데이터 경로
test_file = './data/val_data.csv'  # 검증 데이터 경로
model_name = 'dt'  # 모델 기본 이름
report_file = f'./reports/{model_name}_validation_report.txt'  # 검증 보고서를 저장할 파일 경로


# 파이프라인 실행
main(train_file, test_file, model_name, report_file)
