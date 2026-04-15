"""
preprocess.py
-------------
원본 CSV 파싱 → 시간 집계 → 결측 보간 → 스케일링 → Train/Test 분할

설계 근거:
- 쉼표가 유럽식 소수점 구분자로 사용되었으며, 일부 컬럼(pH, Density)은
  소수점 위치가 1~2자리 밀린 포맷 오류가 존재함
- 도메인 물리 범위를 기준으로 파싱 후 배율 보정(/10, /100, /1000)을 시도하는
  방식으로 규칙 기반 정제를 수행함
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ===========================================================
# 각 변수의 물리적 허용 범위 (공정 도메인 지식 기반)
# 이 범위를 이용해 쉼표 해석 방식(소수점 vs 배율오류)을 판별함
# ===========================================================
DOMAIN_RANGES = {
    '% Iron Feed':                  (40.0,  75.0),
    '% Silica Feed':                (1.0,   30.0),
    'Starch Flow':                  (0.0,   6000.0),
    'Amina Flow':                   (0.0,   1500.0),
    'Ore Pulp Flow':                (200.0, 700.0),
    'Ore Pulp pH':                  (8.0,   12.0),
    'Ore Pulp Density':             (1.2,   2.2),
    'Flotation Column 01 Air Flow': (100.0, 400.0),
    'Flotation Column 02 Air Flow': (100.0, 400.0),
    'Flotation Column 03 Air Flow': (100.0, 400.0),
    'Flotation Column 04 Air Flow': (100.0, 400.0),
    'Flotation Column 05 Air Flow': (100.0, 400.0),
    'Flotation Column 06 Air Flow': (100.0, 400.0),
    'Flotation Column 07 Air Flow': (100.0, 400.0),
    'Flotation Column 01 Level':    (100.0, 900.0),
    'Flotation Column 02 Level':    (100.0, 900.0),
    'Flotation Column 03 Level':    (100.0, 900.0),
    'Flotation Column 04 Level':    (100.0, 900.0),
    'Flotation Column 05 Level':    (100.0, 900.0),
    'Flotation Column 06 Level':    (100.0, 900.0),
    'Flotation Column 07 Level':    (100.0, 900.0),
    '% Iron Concentrate':           (55.0,  72.0),
    '% Silica Concentrate':         (0.5,   8.0),
}


def parse_with_domain(raw, col_min: float, col_max: float) -> float:
    """
    유럽식 쉼표 소수점을 올바른 float로 변환한다.

    전략:
    1. 마지막 쉼표를 소수점으로 치환하여 candidate 생성
    2. 도메인 범위 내에 있으면 즉시 반환
    3. 범위 이탈 시 /10 → /100 → /1000 순으로 배율 보정 탐색
       (Ore Pulp pH는 /10, Density는 /100 보정이 필요한 것이 실측으로 확인됨)
    4. 어떤 배율로도 범위 내 진입 불가 시 NaN 반환
    """
    s = str(raw).strip()
    if s in ('', 'nan', 'None'):
        return np.nan

    if ',' in s:
        left, right = s.rsplit(',', 1)
        try:
            val = float(left.replace(',', '') + '.' + right)
        except ValueError:
            return np.nan
    else:
        try:
            val = float(s)
        except ValueError:
            return np.nan

    if col_min <= val <= col_max:
        return val

    for divisor in [10.0, 100.0, 1000.0]:
        adj = val / divisor
        if col_min <= adj <= col_max:
            return adj

    return np.nan


def load_and_parse(file_path: str) -> pd.DataFrame:
    """원본 CSV를 로드하고 도메인 기반 파싱을 적용한다."""
    df_raw = pd.read_csv(file_path)
    parsed = pd.DataFrame()
    parsed['date'] = pd.to_datetime(df_raw['date'])
    for col, (lo, hi) in DOMAIN_RANGES.items():
        parsed[col] = df_raw[col].apply(lambda x: parse_with_domain(x, lo, hi))
    print(f"[load_and_parse] 원본 {len(parsed)}행 파싱 완료")
    return parsed


def aggregate_hourly(parsed_df: pd.DataFrame) -> pd.DataFrame:
    """
    15초 단위 원본을 1시간 단위로 평균 집계한다.
    NaN이 포함된 개별 값은 평균 계산에서 자동 제외(skipna=True 기본값).
    """
    num_cols = list(DOMAIN_RANGES.keys())
    averaged = (
        parsed_df
        .groupby('date')[num_cols]
        .mean()
        .reset_index()
        .sort_values('date')
        .reset_index(drop=True)
    )
    print(f"[aggregate_hourly] 시간 집계 후 {len(averaged)}행")
    return averaged


def interpolate_missing(averaged_df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """
    선형 보간으로 단기 결측 구간을 복원한다.
    limit 매개변수: 연속 결측 허용 최대 시간 수
    limit 초과 연속 결측 → NaN 유지 → 이후 dropna로 제거

    이유: LSTM은 시계열 연속성을 가정하므로, 장기 공백 구간을
          임의 보간하면 오히려 학습 신호를 오염시킴.
    """
    num_cols = list(DOMAIN_RANGES.keys())
    df = averaged_df.copy()
    df[num_cols] = df[num_cols].interpolate(
        method='linear', limit=limit, limit_direction='forward'
    )
    before = len(df)
    df = df.dropna(subset=num_cols).reset_index(drop=True)
    after = len(df)
    print(f"[interpolate_missing] 결측 제거: {before} → {after}행 ({before - after}행 제거)")
    return df


def split_and_scale(averaged_df: pd.DataFrame, seq_len: int = 32,
                    train_ratio: float = 0.7):
    """
    시간 순서를 유지한 채 Train/Test 분할 후 StandardScaler를 적용한다.

    주의:
    - scaler는 반드시 train_df 기준으로만 fit해야 data leakage 방지
    - test_df는 seq_len만큼 overlap을 두어 첫 시퀀스 생성을 보장함
    - target 컬럼도 동일 scaler로 정규화 → 평가 시 inverse_transform 필요
    """
    num_cols = list(DOMAIN_RANGES.keys())
    N = len(averaged_df)
    split_idx = int(N * train_ratio)

    train_df = averaged_df.iloc[:split_idx].reset_index(drop=True)
    test_df  = averaged_df.iloc[split_idx - seq_len:].reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(train_df[num_cols])

    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[num_cols] = scaler.transform(train_df[num_cols])
    test_df[num_cols]  = scaler.transform(test_df[num_cols])

    print(f"[split_and_scale] Train: {len(train_df)}행 / Test: {len(test_df)}행")
    return train_df, test_df, scaler


def run_preprocessing(file_path: str, seq_len: int = 32, train_ratio: float = 0.7):
    """전처리 전 파이프라인 실행 함수."""
    parsed    = load_and_parse(file_path)
    averaged  = aggregate_hourly(parsed)
    cleaned   = interpolate_missing(averaged, limit=3)
    train_df, test_df, scaler = split_and_scale(cleaned, seq_len, train_ratio)
    return train_df, test_df, scaler
