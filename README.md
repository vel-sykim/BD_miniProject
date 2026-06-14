# 🩸 국내 헌혈 수급 현황 분석 및 시계열 기반 헌혈 건수 예측 모델링 프로젝트
> **KOSIS 공공데이터와 머신러닝 기법을 활용한 대한민국 헌혈 실적 및 예측 모델링**

본 프로젝트는 공공데이터포털(KOSIS OpenAPI 및 통계 자료)을 활용하여 지난 20개년(2005년~2024년) 동안의 대한민국 헌혈 데이터(연도별·월별·연령별·직업별·지역별·혈액형별·혈액보유량 등)를 종합적으로 수집·정제하고, 탐색적 데이터 분석(EDA)을 통해 핵심 인사이트를 도출한 뒤, 시계열적 특성을 반영한 선형회귀 및 규제 모델(Ridge, Lasso)을 통해 미래 헌혈 건수를 예측하는 프로젝트입니다.


---


## 📂 프로젝트 핵심 문서 바로가기 (Project Documents)
> 💡 각 아이콘 또는 링크를 클릭하면 해당 문서(노션/구글 드라이브 등)로 바로 이동합니다.

| 분류 | 산출물 목록 | 바로가기 링크 |
| :--- | :--- | :--- |
| **기획 및 제안** | 📑 프로젝트 기획서 & 제안서 | [👉 기획서 1](https://app.notion.com/p/AI-34d1043acf1280938479e01a40e712cf?source=copy_link) /[👉 제안서 2](https://app.notion.com/p/35b1043acf12801d9fe2f434b4c2ea19?source=copy_link) |
| **EDA** | 🧼 데이터 전처리 보고서 | [👉 전처리보고서 보기](https://app.notion.com/p/3651043acf12801db329e69770dcb091?source=copy_link) |
| **데이터 명세서** | 🗂️ 데이터 명세서 (Data Schema) | [👉 데이터명세서 보기](https://app.notion.com/p/35c1043acf1280f9ab6fcfa6cf096538?source=copy_link) |
| **탐색적 분석** | 📊 탐색적 데이터 분석 보고서 (EDA) | [👉 EDA보고서 보기](https://app.notion.com/p/EDA_BD-3591043acf1280e28dd1cffdc03389ce?source=copy_link) |
| **PPT** | 🖨️ 프로젝트 최종 발표 PPT (PDF) | [👉 발표 PPT 보기](https://drive.google.com/file/d/1fJCJ__jedu41jm7nGg6LAQrErPXZKxGg/view?usp=drive_link) |


---


## 🌳 프로젝트 구조 시각화
```text
├── data/                  # 원천 데이터 및 전처리 데이터
├── notebooks/             # jupyter notebook 파일들
│   ├── 01_Data_preprocessing_BD.ipynb
│   ├── 02_EDA_BD.ipynb
│   └── 03_Modeling_BD.ipynb
├── images/                # 시각화 그래프 이미지 저장
└── README.md
```


# 🛠️ 사용 기술 및 개발 환경 (Tech Stack)

Language: Python 3.14+

Data Engineering: Pandas, NumPy, Requests

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn (LinearRegression, Ridge, StandardScaler, VotingRegressor)

# 1. 데이터 API 수집 시스템
# 🌐 공공데이터포털 & KOSIS OpenAPI 기반 데이터 수집 시스템

## 📌 1. 데이터 수집 아키텍처
- **목표**: 매번 사이트에서 CSV를 수동으로 다운로드받는 비효율을 제거하고, 파이썬 `requests` 라이브러리를 활용해 공공데이터포털 및 통계청 KOSIS OpenAPI 서버로부터 직접 20개년 다차원 헌혈 통계 데이터를 수집합니다.
- 
## 🏗️ 파이프라인 아키텍처 요약
1. **Data Ingestion**: 공공데이터포털/KOSIS API를 호출하여 20개년(2005~2024) 다차원 JSON 데이터를 수집 및 데이터프레임화합니다.
2. **Feature Domain Preprocessing**: 분산된 도메인 데이터(연령별, 직업별, 지역별, 재고량)의 노이즈를 제거하고 한글 인코딩 및 타입을 표준화합니다.
3. **Data Leakage Troubleshooting**: 당해 연도 총합 예측 시 미래 정보가 유입되는 현상을 방어하기 위해 모든 연간 통계 피처에 `.shift(1)` 연산을 강제 적용합니다.
4. **Time-Series Matrix Merge**: 년도와 월을 기준으로 가로 병합(Left Join)을 수행하여 최종 머신러닝 입력용 데이터셋을 빌드합니다.

```import os
import requests
import numpy as np
import pandas as pd

# ======================================
# 1. OpenAPI 데이터 수집 (Data Ingestion)
# ======================================
API_KEY = "YOUR_DECODED_OPENAPI_SERVICE_KEY_HERE"
BASE_URL = "http://apis.data.go.kr/1352000/ODMS_STAT_27/w_getOdmsStat27"
os.makedirs('./전처리2', exist_ok=True)

def fetch_blood_data(target_year):
    """지정된 연도의 헌혈 데이터를 OpenAPI로부터 JSON으로 수집 및 DF 변환"""
    params = {'serviceKey': API_KEY, 'pageNo': '1', 'numOfRows': '1000', 'apiType': 'JSON', 'year': str(target_year)}
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        items = response.json().get('response', {}).get('body', {}).get('items', {}).get('item', [])
        return pd.DataFrame(items) if items else None
    except Exception as e:
        print(f"⚠️ {target_year}년 데이터 수집 실패: {e}")
        return None

# 20개년 데이터 자동 수집 배치 가동
collected_frames = [fetch_blood_data(y) for y in range(2005, 2025)]
collected_frames = [df for df in collected_frames if df is not None]

# (샘플 재현용 가이드: API 차단 시 내부 저장된 로컬 CSV 목록을 안전하게 로드하도록 방어 코드 구축)
path = os.path.abspath('./전처리2')
file_map = {'month': 'month2.csv', 'year': 'year2.csv', 'age': 'age2.csv', 'job': 'job2.csv', 'region': 'region2.csv', 'stock': 'stock2.csv'}
data = {}

for key, filename in file_map.items():
    full_path = os.path.join(path, filename)
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        df.columns = df.columns.str.strip()
        data[key] = df

# 개별 도메인 데이터프레임 바인딩 및 컬럼명 표준화
month, year, age, job, region, stock = data['month'], data['year'], data['age'], data['job'], data['region'], data['stock']
for df in [month, year, age, job, region, stock]:
    df.rename(columns={'연도': '년도', '기준연도': '년도', '기준년도': '년도'}, errors='ignore', inplace=True)
    
month['년도'] = month['년도'].astype(int)
month['월'] = month['월'].astype(int)

# ======================================
# 2. 도메인별 범주형 데이터 전처리 및 피벗 연산 (Feature Engineering)
# ======================================
# [연령별] 불필요 합계 제거 및 연령대 매핑 정제
age = age[age["연령코드"] != "A001"]
age_map = {"A002": "16~19세", "A003": "20~29세", "A004": "30~39세", "A005": "40~49세", "A006": "50~59세", "A007": "60세이상"}
age["연령대"] = age["연령코드"].map(age_map)
age_pivot = age.pivot_table(index='년도', columns='연령대', values='헌혈건수', aggfunc='sum').add_prefix('age_')

# [직업별] 결측 기호 기계학습용 수치(0)로 대체
job['건'] = job['건'].replace('-', np.nan).fillna(0).astype(int)
job_pivot = job.pivot_table(index='년도', columns='직업명', values='헌혈건수', aggfunc='sum').add_prefix('job_')

# [지역별] 시도 단위 피벗 집계
region_pivot = region.pivot_table(index='년도', columns='시도명', values='헌혈건수', aggfunc='sum').add_prefix('region_')

# [재고량] 수치형 정제 및 그룹화
stock_clean = stock.groupby('년도').sum(numeric_only=True).add_prefix('stock_')

# ========================================
# 3. 데이터 누수 차단 (Data Leakage Troubleshooting) & 시계열 병합
# ========================================
# 🚨핵심 가치: 금년도 총헌혈건수 예측 시, 금년도 세부 통계가 유입되는 반칙(Leakage)을 차단합니다.
# 모든 연간 통계 및 재고 지표를 .shift(1) 하여 "전년도 마감 실적" 정보만 피처로 사용하도록 시간 인과 관계를 수정합니다.
age_pivot_lagged = age_pivot.shift(1)
job_pivot_lagged = job_pivot.shift(1)
region_pivot_lagged = region_pivot.shift(1)
year_lagged = year.set_index('년도').shift(1)
stock_lagged = stock_clean.shift(1)

# 시계열 월별 베이스 테이블에 전년도 지표들을 순차적으로 Left Join 병합
merged = pd.merge(month, year_lagged, left_on='년도', right_index=True, how='left')
merged = pd.merge(merged, age_pivot_lagged, left_on='년도', right_index=True, how='left')
merged = pd.merge(merged, job_pivot_lagged, left_on='년도', right_index=True, how='left')
merged = pd.merge(merged, region_pivot_lagged, left_on='년도', right_index=True, how='left')
merged = pd.merge(merged, stock_lagged, left_on='년도', right_index=True, how='left')

# 시프트 연산으로 인해 과거 정보가 없는 최초 1개년(2005년) 데이터 행 제거 및 결측치 정제
merged.dropna(subset=[col for col in merged.columns if col not in ['년도', '월', '총헌혈건수']], inplace=True)
merged.fillna(0, inplace=True)

# 시간 흐름에 따른 정렬 최적화 및 인덱스 초기화
merged = merged.sort_values(['년도', '월']).reset_index(drop=True)

# =======================================
# 4. 머신러닝 주입용 최종 파일 익스포트
# =======================================
output_file = 'merged_BD.csv'
merged.to_csv(output_file, index=False, encoding='utf-8-sig')

print("\n" + "="*60)
print(f"🏆 [파이프라인 빌드 완료] 최종 학습용 데이터셋이 안전하게 결합되었습니다.")
print(f"▶️ 파일 저장 경로: {os.path.abspath(output_file)}")
print(f"▶️ 데이터셋 매트릭스 구조: {merged.shape} (행: {merged.shape[0]}개월분 | 열: {merged.shape[1]}개 피처)")
print("="*60)
```




# 📊 2. 탐색적 데이터 분석 (EDA) 핵심 인사이트

지표 다각화 분석을 통해 국내 혈액 수급 시장의 중장기적 위험 요소를 포착했습니다.


<img width="1690" height="990" alt="image" src="https://github.com/user-attachments/assets/a26c599e-d0dc-4a11-bd98-95a8a4cfaecd" />
1.1 참여 충성도 분석 (기존 참여자 의존도 심화)
신규 헌혈 유입 인구는 급격히 감소하는 반면, 1인당 평균 헌혈 빈도는 지속적으로 강한 상승세를 보입니다. 즉, 기존 충성 고관여 참여자들에게 공급 전체가 강하게 의존하고 있는 불균형 구조를 확인했습니다.

<img width="723" height="526" alt="image-3" src="https://github.com/user-attachments/assets/26955a51-70cd-4311-8a11-2e2a96b6b9d1" />

1.2 연령대별 참여 분포 (20대 주도성 확인)국내 헌혈 실적의 최대 핵심 엔진은 20대 집단이며, 가장 강력한 참여 볼륨을 형성하고 있습니다.

<img width="841" height="504" alt="image-2" src="https://github.com/user-attachments/assets/97adeced-514e-4ce4-9d3a-35eb4aad12d3" />
<img width="1389" height="690" alt="image-1" src="https://github.com/user-attachments/assets/e98eadf3-6e1b-4a89-97d7-fb781eade986" />


1.3 중장기 공급 트렌드 (지속적 하락세)
대한민국 전체 인구구조 변화와 맞물려 중장기 헌혈 실적은 지속적으로 하락하는 우하향 곡선을 그리고 있습니다.


# 🤖 2. 머신러닝 트러블슈팅 서사 (Troubleshooting)
본 프로젝트의 진정한 가치는 시계열 예측 연산 과정에서 마주한 데이터 결함 요인들을 분석하고, 단계별 규제를 통해 극복(Troubleshooting)해 나간 스토리에 있습니다.

🚨 [Issue 1] 초기 선형 모델의 과적합 및 데이터 누수 (Data Leakage)
- 문제 상황: 초기 모델 검증 결과 Train R^2 스코어가 정확히 1.0000이 산출되는 모순 발생.
- 원인 분석: 당해 연도 총합을 예측하는 타겟 컬럼에 당해 연도의 세부 구성 지표들이 가로 병합 피처로 동시에 입력되면서 미래 정보가 모델 내부로 유출된 물리적 왜곡 구조 확인.
- 해결 조치: 모든 연간 거시 통계 데이터 및 재고 데이터 스키마에 .shift(1) 가공 연산을 강제 전제하여, 금년도 예측 시에는 무조건 '전년도 전체 실적'만 학습 독립변수로 활용하도록 시계열 인과 구조 재수립.

🔬 [Issue 2] 특정 알고리즘군의 예측 파괴 현상 규명누수 차단 후 다양한 고도화 모델을 적용하였으나, 특정 모델의 스코어가 마이너스로 폭망하거나 0 근처에 정체하는 치명적인 왜곡 발생.
- Lasso 모델의 붕괴 (Test\ R2: -1.0593$): 상관성이 밀집된 다차원 도메인 피처 공간에서 가중치를 강제로 $0$으로 탈락시키는 Lasso의 특성이 모델의 일반화 복원력을 완전 무력화함.
- XGBoost 모델의 무력화 (Test\ R2: 0.0338$): 트리 기반 분할 알고리즘은 Train 데이터셋의 수치 범위를 넘어서는 장기적 시계열 트렌드를 마주했을 때 예측 능력을 잃어버리는 외삽(Extrapolation) 한계가 존재함이 실증적으로 판명됨

🏆 [Solution] 최적의 선형-규제 앙상블을 통한 최종 타결
데이터의 본질적인 선형 트렌드를 유연하게 유지하면서도 다중공선성을 완화할 수 있도록 가중치 분산 제어력이 입증된 Linear Regression과 Ridge 모델을 최종 타겟 유니온으로 선정했습니다.
성능을 좀먹던 Lasso와 XGBoost를 앙상블 레이어에서 완전히 걷어내고, 안정적인 Ridge 모델에 높은 가중치를 배정한 VotingRegressor(weights=[1, 2]) 구조를 정립하여 역대 최고 수준의 일반화 스코어를 갱신했습니다.

# <code> 데이터 로드 및 수치형 피처 정제
```import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

## 1. 데이터 로드
merged = pd.read_csv('merged_BD.csv')

## 2. 타겟 및 피처 분리
y = merged['총헌혈건수']

## 🚨 [고도화 포인트] 선형 모델이 '년도/월'을 단순 연속형 숫자로 오인하여 
## 미래 가중치를 왜곡하는 현상을 방지하기 위해 피처 공간에서 완전 배제합니다.
X = merged.drop(columns=['총헌혈건수', '년도', '월'], errors='ignore')
X = X.select_dtypes(include=[np.number])

print(f"✅ 학습 피처 구조: {X.shape} | 타겟 구조: {y.shape}")

## 📅 2. 엄격한 시계열 검증 전략 (Time-Series Split)
- 일반적인 `train_test_split(shuffle=True)`을 사용할 경우, 시계열 데이터의 과거와 미래가 무작위로 섞여 무의미한 미래 예측 모델이 됩니다.
- 본 모델링에서는 **데이터의 시간적 순서를 철저히 보존하기 위해 순차적 8:2 분리 방식**을 채택하여 일반화 성능을 검증합니다.
## 순차적 80% 지점 계산
split_idx = int(len(merged) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

## 선형 모델의 안정적 수렴을 위한 StandardScaler 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"📦 Train 셋 기간: {len(X_train)}개월 | Test 셋 기간: {len(X_test)}개월")

## 최종모델 학습 및 평가
## 모델 정의
best_lr = LinearRegression()
best_ridge = Ridge(alpha=10.0, random_state=42)

## 가중치 앙상블 모델 선언
final_voting = VotingRegressor(
    estimators=[('lr', best_lr), ('ridge', best_ridge)],
    weights=[1, 2],
    n_jobs=-1
)

## 학습 및 평가
final_voting.fit(X_train_scaled, y_train)
tr_pred = final_voting.predict(X_train_scaled)
te_pred = final_voting.predict(X_test_scaled)

print("===== 🏆 최종 고도화 Voting Regressor 결과 =====")
print(f"Train R² Score (훈련 설명력) : {r2_score(y_train, tr_pred):.4f}")
print(f"Test R² Score  (테스트 설명력) : {r2_score(y_test, te_pred):.4f}")
print(f"MAE            (평균 절대 오차) : {mean_absolute_error(y_test, te_pred):.2f} 건")
print(f"RMSE           (제곱근 오차)    : {np.sqrt(mean_squared_error(y_test, te_pred)):.2f} 건")
```



# 📈 3. 최종 모델 성능 평가 지표 (Model Results)모든 데이터 누수 차단 및 스케일링 통일을 거친 최종 융합 앙상블 모델의 스코어 정보입니다.

Train R^2 Score (훈련 데이터 설명력): 0.9999Test 
R^2 Score (테스트 데이터 설명력): 0.9990
MAE (평균 절대 오차): 1,806.31
RMSE (평균 제곱근 오차): 2,229.41

💡 최종 결론: 미래 데이터 원천 차단 시나리오 하에서도 $99.9%$의 테스트 데이터 설명력이 확보된다는 것은, 전년도의 인구학적 세부 통계 변화 지표가 차년도 수급 변동성을 안정적으로 바인딩하는 최고의 선형 지표임을 통계 및 머신러닝으로 증명해낸 고가치 성과입니다.


## 📝 6. 회고 및 향후 발전 방향 (Retrospective)

### 6.1 프로젝트 성과 및 배운 점
* **End-to-End 데이터 파이프라인 자립 구축:** 공공 KOSIS Open API를 활용한 로우 데이터 수집부터 다차원 테이블 피벗, 시계열 결합(Merge), 그리고 규제 기반 앙상블 모델링까지 머신러닝의 전 과정을 외부 도움 없이 독립적으로 설계하며 데이터 엔지니어링 역량을 크게 내재화했습니다.
* **통계적 직관과 트러블슈팅 역량 확보:** Train R2 점수가 1.0 이 나오는 비정상적인 상황을 마주했을 때, 모델의 정답 유출(Data Leakage)을 의심하고 `.shift(1)` 전략을 통해 인과 구조를 바로잡는 진짜 엔지니어링적 디버깅 과정을 경험했습니다.
* **알고리즘의 한계와 특성 실증:** 단순히 유행하는 딥러닝이나 복잡한 트리 모델을 맹신하기보다, 데이터의 선형적 본질을 파악하고 알고리즘 특성(XGBoost의 외삽 한계, Lasso의 다중공선성 취약성)에 맞춰 최적의 솔루션을 찾아내는 '데이터 중심적 사고'의 중요성을 체득했습니다.

### 6.2 현재 시스템의 한계점 분석
* **내부 실적 데이터에 대한 높은 의존도:** 현재 예측 엔진은 전년도의 인구학적 헌혈 실적 피처에 강하게 의존하고 있습니다. 실제 국내 혈액 수급량은 계절성 유행 질환(감기, 독감 등), 국가적 보건 위기(팬데믹), 군부대 훈련 일정 및 헌혈 장려 캠페인 같은 외부 거시적 환경 요인에 민감하게 반응하지만, 현재 파이프라인에는 해당 외부 변수가 반영되지 못했습니다.
* **데이터 샘플 사이즈의 제한:** 연간 및 월간 단위로 결합된 시계열 데이터 특성상, 전체 학습 샘플의 수가 머신러닝 기준으로는 다소 제한적입니다. 이는 데이터가 가진 장기적 트렌드는 완벽히 설명하지만, 돌발적인 공급 충격(Shock)을 방어하는 일반화 성능 면에서는 리스크가 존재할 수 있습니다.

### 6.3 향후 고도화 계획 (Future Work)
* **비선형 외부 변수를 포함한 하이브리드(Hybrid) 모델로의 확장:** 향후 거시적 보건 지표, 계절성 날씨 변수, 인구 감소율 데이터를 수집하여 파이프라인에 추가할 예정입니다. 이후 데이터의 거대한 선형 트렌드는 본 프로젝트에서 검증된 `Ridge` 모델이 정밀하게 바인딩하고, 미세한 비선형 잔차(Residual) 오차는 `XGBoost`나 `LightGBM`이 정교하게 잡아내도록 설계하는 **시계열 하이브리드 예측 엔진** 구축을 최종 목표로 하고 있습니다.
