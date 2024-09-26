import os
import pandas as pd

"""
이 코드는 주어진 CSV 파일에서 MAC 주소를 B1, B2, ..., B18로 매핑하여 데이터를 변환한 후,
TimeStamp와 Zone을 기준으로 피벗 테이블을 생성하여 각 MAC 주소에 해당하는 RSSI 값을 열로 변환하는 작업을 수행합니다.
최종적으로 변환된 데이터를 새로운 CSV 파일로 저장합니다.

주요 기능:
1. 특정 MAC 주소를 B1-B18로 매핑
2. 입력 폴더 내의 CSV 파일들을 읽어와, MAC 주소를 B1-B18로 변환
3. TimeStamp와 Zone을 기준으로 피벗 테이블을 생성해, 각 B1-B18에 해당하는 RSSI 값을 열로 변환
4. NaN 값 또는 누락된 B1-B18 열을 0으로 채우고, 모든 값을 정수형으로 변환
5. 변환된 데이터를 출력 폴더에 저장

"""
# 특정 MAC 주소를 B1, B2, ..., B18로 매핑하는 딕셔너리 정의
mac_to_b_columns = {
    '60:98:66:33:42:D4': 'B1',
    '60:98:66:32:8E:28': 'B2',
    '60:98:66:32:BC:AC': 'B3',
    '60:98:66:30:A9:6E': 'B4',
    '60:98:66:32:CA:74': 'B5',
    '60:98:66:2F:CF:9F': 'B6',
    '60:98:66:32:B8:EF': 'B7',
    '60:98:66:32:CA:59': 'B8',
    '60:98:66:33:35:4C': 'B9',
    '60:98:66:32:AF:B6': 'B10',
    '60:98:66:33:0E:8C': 'B11',
    '60:98:66:32:C8:E9': 'B12',
    '60:98:66:32:9F:67': 'B13',
    '60:98:66:33:24:44': 'B14',
    '60:98:66:32:BB:CB': 'B15',
    '60:98:66:32:AA:F8': 'B16',
    'A0:6C:65:99:DB:7C': 'B17',
    '60:98:66:32:98:58': 'B18'
}

def transform_csv_files(input_folder_path, output_folder_path):
    # 출력 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 입력 폴더 내의 모든 CSV 파일 목록 가져오기
    csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]

    # 각 CSV 파일을 순차적으로 처리
    for file_name in csv_files:
        # CSV 파일 읽기
        file_path = os.path.join(input_folder_path, file_name)
        data = pd.read_csv(file_path)
        
        # 데이터프레임에서 MAC 주소를 B1, B2, ..., B18로 변환
        data['B_column'] = data['MAC Address'].map(mac_to_b_columns)

        # 피벗 테이블로 변환하여 각 TimeStamp마다 B1-B18을 별도 열로 설정
        pivoted_data = data.pivot_table(index=['TimeStamp', 'Zone'], columns='B_column', values='RSSI').reset_index()

        # 열 순서를 TimeStamp, B1-B18, Zone 순으로 재정렬
        columns = ['TimeStamp'] + [f'B{i}' for i in range(1, 19)] + ['Zone']
        for b_col in columns[1:-1]:
            if b_col not in pivoted_data.columns:
                pivoted_data[b_col] = 0  # B1-B18 열이 존재하지 않는 경우, 0으로 채우기

        # NaN 값을 0으로 채우기
        pivoted_data = pivoted_data.fillna(0)

        # RSSI 값을 정수형으로 변환
        for b_col in columns[1:-1]:
            pivoted_data[b_col] = pivoted_data[b_col].astype(int)

        # 올바른 열 순서로 정렬
        pivoted_data = pivoted_data[columns]

        # 변환된 데이터를 출력 폴더에 새로운 CSV 파일로 저장
        output_file_path = os.path.join(output_folder_path, f"transformed_{file_name}")
        pivoted_data.to_csv(output_file_path, index=False)

    print(f"모든 CSV 파일이 변환되어 {output_folder_path}에 저장되었습니다.")

# 사용 예시:
# 실제 입력 폴더 경로와 출력 폴더 경로를 지정하세요
input_folder_path = './data/_test_dataset'  # 실제 입력 폴더 경로로 업데이트
output_folder_path = './data/_test_dataset_transformed'  # 실제 출력 폴더 경로로 업데이트
transform_csv_files(input_folder_path, output_folder_path)
