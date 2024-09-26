from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # CORS 미들웨어 설정

socketio = SocketIO(app, cors_allowed_origins="*")  # 모든 출처 허용

# 저장된 머신러닝 모델과 레이블 인코더 불러오기
model = joblib.load('./_0_RF_REF(X)_shuffle(O)_model.pkl')  # 모델 파일 경로
label_encoder = joblib.load('./_0_RF_REF(X)_shuffle(O)_label_encoder.pkl')  # 레이블 인코더 경로

@app.route('/')
def index():
    return 'WebSocket server is running!'

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('message')
def handle_message(message):
    try:
        # 받은 메시지를 JSON 형식으로 파싱
        data = json.loads(message)
        print('Received data:', data)

        # 비콘 데이터 처리
        if isinstance(data, list) and len(data) > 0:
            beacon_data = data[0]  # 첫 번째 데이터를 사용
            print(f"Timestamp: {beacon_data['TimeStamp']}")

            # 받은 비콘 데이터를 DataFrame으로 변환
            beacon_values = {f'B{i}': beacon_data[f'B{i}'] for i in range(1, 19)}  # B1 ~ B18 값 추출
            df = pd.DataFrame([beacon_values])

            # 머신러닝 모델로 예측 수행
            predicted_zone_encoded = model.predict(df)  # 예측 수행 (모델에 맞게 df 준비)
            predicted_zone = label_encoder.inverse_transform(predicted_zone_encoded)  # 예측된 구역 디코딩

            # 예측 결과를 클라이언트로 전송
            scanner_id = beacon_data.get('scanner_id', 'Unknown')  # scanner_id가 없는 경우 'Unknown' 사용
            result = {
                'scanner_id': scanner_id,
                'floor': 1,  # 예시로 1층
                'zone': predicted_zone[0]  # 예측된 구역 값 전송
            }

            emit('message', json.dumps(result))
            print(f'Sent predicted zone: {result}')
        else:
            print("Received data is not in the expected format")
    except json.JSONDecodeError as e:
        print('Failed to parse message as JSON:', str(e))

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
