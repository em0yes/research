import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

# 모델 이름 설정 
model_name = 'dt'  # 모델 기본 이름

# 테스트 파일들이 있는 폴더 경로
folder_path = "./data/_test_dataset_transformed"

# 예측 결과를 저장할 폴더 경로 (사용자가 설정 가능)
output_folder_path = f"./reports/{model_name}_predicted_results"
os.makedirs(output_folder_path, exist_ok=True)  # 폴더가 없으면 생성

# 저장된 모델과 레이블 인코더 파일 경로
model_file = f'models/{model_name}_model.pkl'
encoder_file = f'models/{model_name}_encoder.pkl'

# 저장된 모델, 레이블 인코더 불러오기
rf = joblib.load(model_file)
le = joblib.load(encoder_file)

# 예측 정확도 및 classification report 결과를 저장할 텍스트 파일 경로
accuracy_output_file = os.path.join(output_folder_path, f"{model_name}_prediction_accuracies.txt")

# 폴더 내 모든 파일에 대해 예측 수행
with open(accuracy_output_file, "w") as accuracy_file:
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            
            # 새로운 데이터 불러오기
            new_data = pd.read_csv(file_path)
            
            # 'Zone' 컬럼을 제외하고, 비콘 값(B1~B18)만 사용
            if 'Zone' in new_data.columns:
                actual_zones = new_data['Zone']  # 실제 Zone 값 저장
                X_new = new_data.drop(columns=['TimeStamp', 'Zone'])  # Zone 제거
            else:
                actual_zones = None
                X_new = new_data.drop(columns=['TimeStamp'])
            
            # 새로운 데이터에 대해 예측
            y_pred_encoded = rf.predict(X_new)

            # 예측된 결과를 다시 문자열로 변환
            y_pred = le.inverse_transform(y_pred_encoded)
            
            # 예측 결과 추가
            new_data['Predicted_Zone'] = y_pred
            
            # 예측 결과를 새로운 파일로 저장
            output_file_path = os.path.join(output_folder_path, f"predicted_{filename}")
            new_data.to_csv(output_file_path, index=False)
            print(f"Predicted results saved as {output_file_path}")
            
            # 정확도 및 classification report 계산 (실제 Zone 값이 존재하는 경우에만)
            if actual_zones is not None:
                accuracy = accuracy_score(actual_zones, y_pred)
                report = classification_report(actual_zones, y_pred, zero_division=1)  # zero_division=1 설정
                
                # 파일에 기록
                accuracy_file.write(f"{filename} - Accuracy: {accuracy:.4f}\n")
                accuracy_file.write(f"Classification Report for {filename}:\n{report}\n")
                print(f"Accuracy for {filename}: {accuracy:.4f}")
                print(f"Classification Report for {filename}:\n{report}")

print("Prediction completed for all files.")
