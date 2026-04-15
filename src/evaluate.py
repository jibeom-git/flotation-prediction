"""
evaluate.py
-----------
저장된 best_model.pth를 로드하여 아래 두 가지 추가 평가를 수행한다.

1. Naive Baseline 비교
   - t-1 시점 값을 그대로 t 시점 예측값으로 사용하는 persistence model
   - 우리 모델이 이 baseline보다 얼마나 더 좋은지 확인

2. Train / Validation / Test 3분할 재평가
   - 기존 2분할(Train 70% / Test 30%)에서
     Train 70% / Val 15% / Test 15%로 변경
   - Val 기준 Early Stopping이 적용됐을 때의 성능을 시뮬레이션

실행 방법:
    python -m src.evaluate --data_path data\MiningProcess_Flotation_Plant_Database.csv
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.preprocess import run_preprocessing, DOMAIN_RANGES, split_and_scale, interpolate_missing, aggregate_hourly, load_and_parse
from src.dataset import FlotationWaveletDataset
from src.model import FlotationNetWithWavelet, weighted_mse

# 기존 train.py와 동일한 설정
FED_PURITY_COLS       = ['% Iron Feed', '% Silica Feed']
PROCESSED_PURITY_COLS = ['% Iron Concentrate', '% Silica Concentrate']
PROCESS_PARAM_COLS    = [
    c for c in DOMAIN_RANGES.keys()
    if c not in FED_PURITY_COLS + PROCESSED_PURITY_COLS
]
MAIN_COLS   = FED_PURITY_COLS + PROCESSED_PURITY_COLS
TARGET_COLS = PROCESSED_PURITY_COLS

SEQ_LEN    = 32
BATCH_SIZE = 64
HPARAMS = {
    'lstm_hidden':     64,
    'lstm_layers':     2,
    'dense_hidden':    64,
    'wavelet_out_dim': 64,
    'wavelet':         'db4',
    'wavelet_level':   2,
}


# =========================================================
# 1. 평가 지표 계산 유틸
# =========================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str):
    """원단위 배열 기준으로 MSE/MAE/R2를 출력한다."""
    print(f"\n  [{label}]")
    for i, name in enumerate(TARGET_COLS):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2  = r2_score(y_true[:, i], y_pred[:, i])
        print(f"    {name:30s} MSE={mse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return


# =========================================================
# 2. Naive Baseline (Persistence Model)
# =========================================================
def naive_baseline(Y_true: np.ndarray):
    """
    t-1 시점의 실제값을 t 시점 예측값으로 사용하는 baseline.
    이보다 우리 모델이 좋아야 실질적인 학습이 이루어진 것.
    """
    y_pred = Y_true[:-1]   # t-1 값을 예측으로 사용
    y_true = Y_true[1:]    # t 값이 정답
    return y_true, y_pred


# =========================================================
# 3. 모델 예측 수집
# =========================================================
@torch.no_grad()
def collect_predictions(model, loader, device, scaler):
    """
    DataLoader에서 예측값을 수집하고 inverse_transform으로 원단위 복원.
    """
    model.eval()
    Ys, YHs = [], []
    for x_seq, x_aux, w_feat, y in loader:
        x_seq, x_aux, w_feat = (t.to(device) for t in (x_seq, x_aux, w_feat))
        y_hat = model(x_seq, x_aux, w_feat)
        Ys.append(y.cpu().numpy())
        YHs.append(y_hat.cpu().numpy())

    Y_scaled  = np.concatenate(Ys,  axis=0)
    YH_scaled = np.concatenate(YHs, axis=0)

    # inverse_transform — target 컬럼 인덱스 기준으로 수동 역변환
    all_cols = list(DOMAIN_RANGES.keys())
    target_idx   = [all_cols.index(c) for c in TARGET_COLS]
    target_mean  = scaler.mean_[target_idx]
    target_scale = scaler.scale_[target_idx]

    Y_orig  = Y_scaled  * target_scale + target_mean
    YH_orig = YH_scaled * target_scale + target_mean
    return Y_orig, YH_orig


# =========================================================
# 4. 3분할 데이터 준비
# =========================================================
def prepare_three_split(cleaned_df, seq_len=SEQ_LEN,
                        train_ratio=0.70, val_ratio=0.15):
    """
    Train 70% / Val 15% / Test 15% 분할.
    scaler는 train 기준으로만 fit.
    """
    from sklearn.preprocessing import StandardScaler

    num_cols = list(DOMAIN_RANGES.keys())
    N = len(cleaned_df)
    train_end = int(N * train_ratio)
    val_end   = int(N * (train_ratio + val_ratio))

    train_df = cleaned_df.iloc[:train_end].reset_index(drop=True)
    val_df   = cleaned_df.iloc[train_end - seq_len:val_end].reset_index(drop=True)
    test_df  = cleaned_df.iloc[val_end - seq_len:].reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(train_df[num_cols])

    for df in [train_df, val_df, test_df]:
        df[num_cols] = scaler.transform(df[num_cols])

    print(f"[3분할] Train={len(train_df)}행 / Val={len(val_df)}행 / Test={len(test_df)}행")
    return train_df, val_df, test_df, scaler


# =========================================================
# 5. 시각화 — 기존 2분할 vs Naive Baseline 비교
# =========================================================
def plot_comparison(Y_true, Y_model, Y_naive_true, Y_naive_pred,
                    results_dir: str):
    for i, name in enumerate(TARGET_COLS):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

        # 모델 예측
        axes[0].plot(Y_true[:200, i], label='True', linewidth=1.5)
        axes[0].plot(Y_model[:200, i], label='Model Pred',
                     linewidth=1.2, linestyle='--')
        axes[0].set_title(f'Model Prediction — {name}')
        axes[0].legend()
        axes[0].grid(True)

        # Naive Baseline
        axes[1].plot(Y_naive_true[:200, i], label='True', linewidth=1.5)
        axes[1].plot(Y_naive_pred[:200, i], label='Naive (t-1)',
                     linewidth=1.2, linestyle='--', color='red')
        axes[1].set_title(f'Naive Baseline (persistence) — {name}')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        fname = name.replace('%', 'pct').replace(' ', '_') + '_comparison.png'
        plt.savefig(os.path.join(results_dir, fname), dpi=150)
        plt.close()
    print(f"[Plot] 비교 그래프 저장 완료")


# =========================================================
# 6. 메인
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='data/MiningProcess_Flotation_Plant_Database.csv')
    parser.add_argument('--model_path', type=str,
                        default='results/best_model.pth')
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 전처리 (기존 2분할) ----
    print("\n" + "=" * 50)
    print("[Step 1] 전처리 (기존 2분할 기준)")
    print("=" * 50)
    train_df, test_df, scaler = run_preprocessing(
        args.data_path, seq_len=SEQ_LEN, train_ratio=0.70
    )

    # ---- Dataset / DataLoader ----
    test_ds = FlotationWaveletDataset(
        test_df, SEQ_LEN, MAIN_COLS, PROCESS_PARAM_COLS, TARGET_COLS,
        wavelet=HPARAMS['wavelet'], level=HPARAMS['wavelet_level']
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, drop_last=False)

    # ---- 모델 로드 ----
    print(f"\n[Step 2] 모델 로드: {args.model_path}")
    w_dim   = test_ds.W_feat.shape[1]
    aux_dim = test_ds.X_aux.shape[1]

    model = FlotationNetWithWavelet(
        aux_dim=aux_dim,
        w_dim=w_dim,
        wavelet_out_dim=HPARAMS['wavelet_out_dim'],
        lstm_hidden=HPARAMS['lstm_hidden'],
        lstm_layers=HPARAMS['lstm_layers'],
        dense_hidden=HPARAMS['dense_hidden'],
        in_channels=len(MAIN_COLS)
    ).to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    print("모델 로드 완료")

    # ---- 기존 2분할 모델 예측 ----
    print("\n" + "=" * 50)
    print("[Step 3] 기존 2분할 Test 성능")
    print("=" * 50)
    Y_true, Y_model = collect_predictions(model, test_loader, device, scaler)
    compute_metrics(Y_true, Y_model, "Model (Train 70% / Test 30%)")

    # ---- Naive Baseline ----
    print("\n" + "=" * 50)
    print("[Step 4] Naive Baseline (Persistence Model)")
    print("=" * 50)
    Y_naive_true, Y_naive_pred = naive_baseline(Y_true)
    compute_metrics(Y_naive_true, Y_naive_pred, "Naive Baseline (t-1 → t)")

    # ---- 개선율 출력 ----
    print("\n" + "=" * 50)
    print("[Step 5] 모델 vs Naive Baseline 개선율")
    print("=" * 50)
    for i, name in enumerate(TARGET_COLS):
        mse_model = mean_squared_error(Y_true[1:, i], Y_model[1:, i])
        mse_naive = mean_squared_error(Y_naive_true[:, i], Y_naive_pred[:, i])
        improvement = (mse_naive - mse_model) / mse_naive * 100
        print(f"  {name:30s} MSE 개선율: {improvement:+.1f}%  "
              f"(Model={mse_model:.4f} / Naive={mse_naive:.4f})")

    # ---- 3분할 재평가 ----
    print("\n" + "=" * 50)
    print("[Step 6] Train / Val / Test 3분할 재평가")
    print("=" * 50)
    parsed   = load_and_parse(args.data_path)
    averaged = aggregate_hourly(parsed)
    cleaned  = interpolate_missing(averaged, limit=3)

    train_df3, val_df3, test_df3, scaler3 = prepare_three_split(cleaned)

    val_ds = FlotationWaveletDataset(
        val_df3, SEQ_LEN, MAIN_COLS, PROCESS_PARAM_COLS, TARGET_COLS,
        wavelet=HPARAMS['wavelet'], level=HPARAMS['wavelet_level']
    )
    test_ds3 = FlotationWaveletDataset(
        test_df3, SEQ_LEN, MAIN_COLS, PROCESS_PARAM_COLS, TARGET_COLS,
        wavelet=HPARAMS['wavelet'], level=HPARAMS['wavelet_level']
    )
    val_loader  = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader3 = DataLoader(test_ds3, batch_size=BATCH_SIZE, shuffle=False)

    Y_val_true,  Y_val_pred  = collect_predictions(model, val_loader,   device, scaler3)
    Y_test_true, Y_test_pred = collect_predictions(model, test_loader3, device, scaler3)

    compute_metrics(Y_val_true,  Y_val_pred,  "Val  (3분할 기준)")
    compute_metrics(Y_test_true, Y_test_pred, "Test (3분할 기준)")

    # ---- 시각화 ----
    print("\n" + "=" * 50)
    print("[Step 7] 비교 시각화 저장")
    print("=" * 50)
    plot_comparison(Y_true, Y_model, Y_naive_true, Y_naive_pred, args.results_dir)

    print("\n평가 완료. 결과는 results/ 폴더를 확인하세요.")


if __name__ == '__main__':
    main()