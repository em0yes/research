import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

def process_folder(input_folder, output_folder, window_size_rows=10, step_size_rows=1):
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

# 윈도우별 평균 계산 후 train_data.csv와 val_data.csv에 저장
def aggregate_window_means_for_training(window_folder, output_file):
    all_csv_files = glob.glob(os.path.join(window_folder, "*.csv"))
    mean_list = []
    
    for file in all_csv_files:
        window_data = pd.read_csv(file)
        # TimeStamp와 Zone을 별도로 처리
        timestamp = window_data['TimeStamp'].iloc[0]  # 첫 번째 TimeStamp를 저장
        zone = window_data['Zone'].iloc[0]  # 첫 번째 Zone을 저장
        
        # TimeStamp와 Zone 제외한 나머지 Beacon 데이터 평균 계산
        beacon_data = window_data.drop(columns=['TimeStamp', 'Zone'])
        beacon_means = beacon_data.replace(0, pd.NA).mean()  # 0을 NaN으로 처리한 후 평균 계산
        beacon_means = beacon_means.fillna(0).round(0).astype(int)  # NaN은 0으로 대체, 반올림 후 int로 변환
        
        # TimeStamp와 Zone을 다시 추가 (TimeStamp를 맨 앞에)
        beacon_means = pd.Series(beacon_means)  # 시리즈로 변환 후 추가
        beacon_means['TimeStamp'] = timestamp
        beacon_means['Zone'] = zone
        
        # 열 순서를 지정해 TimeStamp가 맨 앞에 오도록 설정
        ordered_columns = ['TimeStamp'] + [col for col in beacon_means.index if col != 'TimeStamp']
        beacon_means = beacon_means[ordered_columns]
        
        mean_list.append(beacon_means)
    
    mean_df = pd.DataFrame(mean_list)
    shuffled_mean_df = shuffle(mean_df)
    shuffled_mean_df.to_csv(output_file, index=False)
    print(f"Averaged window data saved to: {output_file}")

if __name__ == "__main__":
    # 1. 슬라이딩 윈도우 처리 및 저장
    input_data_folder = './_dataset_transformed'
    output_window_folder = './processed_sliding_windows'
    process_folder(input_data_folder, output_window_folder, window_size_rows=5, step_size_rows=1)

    # 2. 슬라이딩 윈도우 파일을 train과 val로 나누어 저장
    train_folder = './train_windows'
    val_folder = './val_windows'
    split_and_save_files(output_window_folder, train_folder, val_folder)

    # 3. train 데이터 윈도우의 평균을 구해 저장
    train_file = './train_data.csv'
    aggregate_window_means_for_training(train_folder, train_file)

    # 4. val 데이터 윈도우의 평균을 구해 저장
    val_file = './val_data.csv'
    aggregate_window_means_for_training(val_folder, val_file)
