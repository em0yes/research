
## 1. 데이터셋 구성

### 데이터 전처리 과정
2. **구역 기반 샘플링**: 각 구역(Zone)의 값이 동일한 경우에만 유효한 데이터로 간주합니다.
3. **셔플(Shuffle)**: 데이터를 셔플하여 학습용과 검증용 데이터로 나눕니다.
4. **학습/검증 데이터 분리**: 80%는 학습용, 20%는 검증용 데이터로 사용됩니다.

### 데이터셋 위치
- **Train Data**: `./train_avg_data.csv`
- **Validation Data**: `./val_avg_data.csv`

---

## 2. 알고리즘 학습 성능 실험

### 알고리즘 및 검증 정확도

| Algorithm                | Verification Accuracy | Report                        |
|--------------------------|-----------------------|--------------------------------|
| [Decision Tree](https://github.com/em0yes/research/blob/main/experiments/train_dt.py)           | 0.65                  | [decision_tree_report.txt](https://github.com/em0yes/research/blob/main/experiments/reports/dt_validation_report.txt)     |
| [Random Forest](https://github.com/em0yes/research/blob/main/experiments/train_rf.py)           | 0.76                  | [random_forest_report.txt](https://github.com/em0yes/research/blob/main/experiments/reports/rf_validation_report.txt)     |
| [XGBoost](https://github.com/em0yes/research/blob/main/experiments/train_xgb.py)                | 0.63                  | [xgboost_report.txt](https://github.com/em0yes/research/blob/main/experiments/reports/xgb_validation_report.txt)           |
| [SVM](https://github.com/em0yes/research/blob/main/experiments/train_svm.py)                     | 0.42                  | [svm_report.txt](https://github.com/em0yes/research/blob/main/experiments/reports/svm_validation_report.txt)               |
| [Logistic Regression](https://github.com/em0yes/research/blob/main/experiments/train_lr.py)     | 0.48                  | [logistic_regression_report.txt](https://github.com/em0yes/research/blob/main/experiments/reports/lr_validation_report.txt) |
| [Gradient Boosting Model](https://github.com/em0yes/research/blob/main/experiments/train_gb.py) | 0.64                  | [gradient_boosting_report.txt](https://github.com/em0yes/Research/blob/main/Experiments/reports/gb_validation_report.txt) |
| [KNN](https://github.com/em0yes/research/blob/main/experiments/train_knn.py)                    | 0.59                  | [knn_report.txt](https://github.com/em0yes/research/blob/main/experiments/reports/knn_validation_report.txt)               |

#### 검증 정확도가 가장 높은 Random Forest 알고리즘을 사용하여 최적화를 진행.
---
## 3. 구역별 예측 성능 분석
 
| Algorithm      | Accuracy | A_1  | A_2  | A_3  | B_1  | B_2  | C_1  | C_2  | C_3  | D_1  | D_2  | E_1  | E_2  | E_3  |
| -------------- | -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| [**RF**]         | 0.81     | 56%  | 44%  | 44%  | 23%  | 33%  | 50%  | 73%  | 55%  | 23%  | 18%  | 67%  | 37%  | 58%  |
| [**RF_AVG**]     | 0.91     | 82%  | 98%  | 72%  | 28%  | 75%  | 83%  | 100% | 100% | 55%  | 3%   | 88%  | 60%  | 87%  |
| [**RF_AVG & REF**]| 0.89    | 82%  | 96%  | 74%  | 28%  | 75%  | 92%  | 100% | 100% | 55%  | 0%   | 88%  | 55%  | 79%  |

#### RF_AVG 알고리즘이 윈도우별 평균값을 적용하여 가장 높은 테스트 정확도를 보임. 앞으로 RF_AVG 방식으로 성능을 향상시키는 방향으로 개발 예정.
