import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# 테스트 파일들이 있는 폴더 경로
folder_path = "./_test_dataset_transformed"

# 예측 결과를 저장할 폴더 경로 (사용자가 설정 가능)
output_folder_path = "./_0_RF_shuffle(O)_predicted_results"
os.makedirs(output_folder_path, exist_ok=True)  # 폴더가 없으면 생성

# 저장된 RFE 선택기, 모델, 레이블 인코더 불러오기
# rfe = joblib.load('_0_RF_REF(O)_shuffle(O)_selector.pkl')
rf = joblib.load('_0_RF_shuffle(O)_model.pkl')
le = joblib.load('_0_RF_shuffle(O)_label_encoder.pkl')

# 예측 정확도 및 classification report 결과를 저장할 텍스트 파일 경로
accuracy_output_file = os.path.join(output_folder_path, "prediction_accuracies.txt")

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
            
            # 새로운 데이터에 RFE 적용 (특징 선택)
            # X_new_rfe = rfe.transform(X_new)
            
            # 새로운 데이터에 대해 예측
            # y_pred_encoded = rf.predict(X_new_rfe)
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
            
            # Zone 값을 알파벳과 숫자로 분리하여 정렬
            sorted_zones = new_data['Predicted_Zone'].value_counts().sort_index(key=lambda x: x.str.split('_').map(lambda y: (y[0], int(y[1]))))
            
            # 그래프 1: Predicted Zone Distribution
            plt.figure(figsize=(10, 6))
            sorted_zones.plot(kind='bar', color='skyblue')
            plt.title(f'Predicted Zone Distribution for {filename}')
            plt.xlabel('Zone')
            plt.ylabel('Count')
            
            # 그래프 저장
            graph_output_path = os.path.join(output_folder_path, f"graph_{filename.replace('.csv', '.png')}")
            plt.savefig(graph_output_path)
            plt.close()  # 그래프를 파일로 저장한 후 닫음
            print(f"Graph saved as {graph_output_path}")

            # 그래프 2: Timestamp vs Predicted Zone
            plt.figure(figsize=(12, 6))
            
            # Zone 값을 알파벳과 숫자 순서로 정렬
            sorted_zone_order = sorted(new_data['Predicted_Zone'].unique(), key=lambda x: (x.split('_')[0], int(x.split('_')[1])))

            # Zone을 숫자로 맵핑하여 시각화 (y축은 Zone 값 대신 정렬된 순서로 매핑)
            zone_mapping = {zone: i for i, zone in enumerate(sorted_zone_order)}
            new_data['Zone_Mapped'] = new_data['Predicted_Zone'].map(zone_mapping)
            
            plt.plot(new_data['TimeStamp'], new_data['Zone_Mapped'], marker='o', linestyle='-', color='orange')
            plt.xticks(rotation=45)
            plt.yticks(ticks=range(len(sorted_zone_order)), labels=sorted_zone_order)  # y축에 정렬된 Zone 값 표시
            plt.title(f'Timestamp vs Predicted Zone for {filename}')
            plt.xlabel('Timestamp')
            plt.ylabel('Predicted Zone')
            
            # 두 번째 그래프 저장
            timestamp_graph_output_path = os.path.join(output_folder_path, f"timestamp_graph_{filename.replace('.csv', '.png')}")
            plt.savefig(timestamp_graph_output_path)
            plt.close()
            print(f"Timestamp graph saved as {timestamp_graph_output_path}")

print("Prediction and graph generation completed for all files.")
