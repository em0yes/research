import pandas as pd
import os
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

class ZoneClassifierWithSlidingWindow:
    def __init__(self, model, window_size_rows=5, step_size_rows=5):
        # 다양한 모델을 초기화할 수 있도록 수정
        self.model = model
        self.window_size = window_size_rows  # 슬라이딩 윈도우 크기 (데이터 개수 기준)
        self.step_size_rows = step_size_rows  # 슬라이딩 윈도우 이동 간격 (데이터 개수 기준)

    def load_data(self, train_file, test_file):
        """
        학습 및 검증 데이터를 CSV 파일에서 불러옵니다.
        """
        self.train_data = pd.read_csv(train_file)
        self.val_data = pd.read_csv(test_file)

        # Zone 열을 Label Encoding 처리
        self.label_encoder = LabelEncoder()
        self.train_data['Zone'] = self.label_encoder.fit_transform(self.train_data['Zone'])
        self.val_data['Zone'] = self.label_encoder.transform(self.val_data['Zone'])

    def preprocess_data(self, data):
        """
        데이터를 슬라이딩 윈도우 방식으로 그룹화하여 반환합니다.
        """
        data = data.sort_values(by='TimeStamp')

        grouped_data = []
        grouped_labels = []
        start_index = 0  # 첫 시작 index

        # 슬라이딩 윈도우 방식으로 그룹화
        while start_index + self.window_size <= len(data):
            window_data = data.iloc[start_index:start_index + self.window_size]

            # 윈도우 내 Zone 값이 모두 동일한지 확인
            if not window_data.empty and window_data['Zone'].nunique() == 1:
                window_data_no_zero = window_data.replace(0, pd.NA)
                grouped_data.append(window_data_no_zero.drop(columns=['TimeStamp', 'Zone']).mean(skipna=True).tolist())
                grouped_labels.append(window_data['Zone'].iloc[-1])

            start_index += self.step_size_rows

        return pd.DataFrame(grouped_data), grouped_labels

    def train_model(self):
        """
        학습 데이터를 통해 모델을 학습합니다.
        """
        self.X_train, self.y_train = self.preprocess_data(self.train_data)
        self.model.fit(self.X_train, self.y_train)
        print("모델 학습 완료")

    def validate_model(self):
        """
        검증 데이터를 통해 모델을 검증하고 성능을 출력합니다.
        """
        self.X_val, self.y_val = self.preprocess_data(self.val_data)
        y_pred = self.model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        print(f"검증 정확도: {accuracy:.2f}")
        print("분류 보고서:")
        print(classification_report(self.y_val, y_pred))

    def save_model(self, model_file):
        """
        학습된 모델을 파일로 저장합니다.
        """
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"모델이 {model_file}에 저장되었습니다.")


def main(train_file, test_file, model_file, models):
    """
    여러 모델을 학습하고 성능을 비교합니다.
    """
    for model_name, model in models:
        print(f"\n모델 테스트: {model_name}")
        classifier = ZoneClassifierWithSlidingWindow(model=model, window_size_rows=5, step_size_rows=1)

        # 1단계: 데이터를 불러오기
        classifier.load_data(train_file, test_file)

        # 2단계: 학습 모델 학습
        classifier.train_model()

        # 3단계: 모델 검증 및 성능 확인
        classifier.validate_model()

        # 4단계: 학습된 모델 저장
        classifier.save_model(f'{model_file}_{model_name}.pkl')


if __name__ == '__main__':
    # 학습 및 검증 데이터 경로 설정
    train_file = './data/train_data.csv'
    test_file = './data/val_data.csv'
    model_file = './models/model'

    # 여러 모델 정의
    models = [
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('SVM', SVC(kernel='linear', random_state=42)),
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000)),
        ('NaiveBayes', GaussianNB()),
        ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=42))
    ]

    # 파이프라인 실행
    main(train_file, test_file, model_file, models)
