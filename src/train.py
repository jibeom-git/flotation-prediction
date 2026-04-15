"""
train.py
--------
전처리 → Dataset → 학습 → 평가 → 결과 저장 전 과정을 실행하는 메인 스크립트

실행 방법:
    python src/train.py --data_path data/MiningProcess_Flotation_Plant_Database.csv
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocess import run_preprocessing, DOMAIN_RANGES
from dataset import FlotationWaveletDataset
from model import FlotationNetWithWavelet, weighted_mse


# =========================================================
# 하이퍼파라미터
# =========================================================
HPARAMS = {
    'seq_len':         32,
    'train_ratio':     0.7,
    'batch_size':      64,
    'epochs':          200,
    'lr':              1e-3,
    'weight_decay':    1e-4,   # L2 정규화 (Adam에 포함)
    'lstm_hidden':     64,
    'lstm_layers':     2,
    'dense_hidden':    64,
    'wavelet_out_dim': 64,
    'wavelet':         'db4',
    'wavelet_level':   2,
    'patience':        20,     # Early Stopping 인내 에폭 수
    'grad_clip':       1.0,    # Gradient Clipping 임계값
    'lam1':            0.3,    # Fe 손실 가중치
    'lam2':            0.7,    # Si 손실 가중치
}

# 컬럼 정의
FED_PURITY_COLS       = ['% Iron Feed', '% Silica Feed']
PROCESSED_PURITY_COLS = ['% Iron Concentrate', '% Silica Concentrate']
PROCESS_PARAM_COLS    = [
    c for c in DOMAIN_RANGES.keys()
    if c not in FED_PURITY_COLS + PROCESSED_PURITY_COLS
]
MAIN_COLS   = FED_PURITY_COLS + PROCESSED_PURITY_COLS
TARGET_COLS = PROCESSED_PURITY_COLS


def build_loaders(train_df, test_df):
    """Dataset 및 DataLoader를 생성한다."""
    hp = HPARAMS
    print("\n[Dataset] Train Dataset 생성 중...")
    train_ds = FlotationWaveletDataset(
        train_df, hp['seq_len'], MAIN_COLS, PROCESS_PARAM_COLS, TARGET_COLS,
        wavelet=hp['wavelet'], level=hp['wavelet_level']
    )
    print("[Dataset] Test Dataset 생성 중...")
    test_ds = FlotationWaveletDataset(
        test_df, hp['seq_len'], MAIN_COLS, PROCESS_PARAM_COLS, TARGET_COLS,
        wavelet=hp['wavelet'], level=hp['wavelet_level']
    )

    train_loader = DataLoader(train_ds, batch_size=hp['batch_size'],
                              shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=hp['batch_size'],
                              shuffle=False, drop_last=False)

    return train_loader, test_loader, train_ds


def build_model(train_ds):
    """모델을 초기화하고 device로 이동한다."""
    hp = HPARAMS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Model] Device: {device}")

    w_dim   = train_ds.W_feat.shape[1]
    aux_dim = train_ds.X_aux.shape[1]

    model = FlotationNetWithWavelet(
        aux_dim=aux_dim,
        w_dim=w_dim,
        wavelet_out_dim=hp['wavelet_out_dim'],
        lstm_hidden=hp['lstm_hidden'],
        lstm_layers=hp['lstm_layers'],
        dense_hidden=hp['dense_hidden'],
        in_channels=len(MAIN_COLS)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 학습 가능 파라미터: {total_params:,}개")

    return model, device


def train(model, train_loader, test_loader, device, results_dir: str):
    """
    학습 루프.
    - Gradient Clipping으로 LSTM의 gradient explosion 방지
    - ReduceLROnPlateau로 test loss 정체 시 LR 감소
    - Early Stopping으로 과적합 시 조기 종료
    - Best 모델 state를 results/ 에 저장
    """
    hp = HPARAMS
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, verbose=True
    )

    best_test_loss   = float('inf')
    best_state       = None
    patience_counter = 0
    epoch_losses, test_epoch_losses = [], []

    print("\n[Train] 학습 시작...")
    for epoch in range(1, hp['epochs'] + 1):
        # ---- Train ----
        model.train()
        train_loss, n = 0.0, 0
        for x_seq, x_aux, w_feat, y in train_loader:
            x_seq, x_aux, w_feat, y = (
                t.to(device) for t in (x_seq, x_aux, w_feat, y)
            )
            optimizer.zero_grad()
            y_hat = model(x_seq, x_aux, w_feat)
            loss = weighted_mse(y_hat, y, hp['lam1'], hp['lam2'])
            loss.backward()

            # Gradient Clipping — LSTM은 exploding gradient에 취약
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp['grad_clip'])

            optimizer.step()
            train_loss += loss.item()
            n += 1
        train_loss /= max(n, 1)

        # ---- Test ----
        model.eval()
        test_loss, m = 0.0, 0
        with torch.no_grad():
            for x_seq, x_aux, w_feat, y in test_loader:
                x_seq, x_aux, w_feat, y = (
                    t.to(device) for t in (x_seq, x_aux, w_feat, y)
                )
                y_hat = model(x_seq, x_aux, w_feat)
                test_loss += weighted_mse(y_hat, y, hp['lam1'], hp['lam2']).item()
                m += 1
        test_loss /= max(m, 1)

        epoch_losses.append(train_loss)
        test_epoch_losses.append(test_loss)
        scheduler.step(test_loss)

        # ---- Best 모델 저장 및 Early Stopping ----
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= hp['patience']:
                print(f"[Train] Early Stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Train {train_loss:.6f} | "
                  f"Test {test_loss:.6f} | Best {best_test_loss:.6f}")

    print(f"[Train] 완료 — Best Test Loss: {best_test_loss:.6f}")

    # Best 모델 복원 및 저장
    model.load_state_dict(best_state)
    ckpt_path = os.path.join(results_dir, 'best_model.pth')
    torch.save(best_state, ckpt_path)
    print(f"[Train] 모델 저장: {ckpt_path}")

    return model, epoch_losses, test_epoch_losses


@torch.no_grad()
def evaluate(model, test_loader, device, scaler, results_dir: str):
    """
    최종 평가 — inverse_transform으로 원단위로 복원 후 MSE/MAE/R2 계산.
    스케일 공간에서의 지표는 실무 해석이 불가능하므로 반드시 역변환 필요.
    """
    model.eval()
    Ys, YHs = [], []
    for x_seq, x_aux, w_feat, y in test_loader:
        x_seq, x_aux, w_feat = (t.to(device) for t in (x_seq, x_aux, w_feat))
        y_hat = model(x_seq, x_aux, w_feat)
        Ys.append(y.cpu().numpy())
        YHs.append(y_hat.cpu().numpy())

    Y_scaled  = np.concatenate(Ys, axis=0)   # (N, 2)
    YH_scaled = np.concatenate(YHs, axis=0)  # (N, 2)

    # --- inverse_transform: target 컬럼 인덱스 추출 ---
    # scaler는 num_cols 전체에 대해 fit되어 있으므로
    # target 위치에 해당하는 mean, scale 값으로 수동 역변환
    all_cols = list(DOMAIN_RANGES.keys())
    target_indices = [all_cols.index(c) for c in TARGET_COLS]
    target_mean  = scaler.mean_[target_indices]
    target_scale = scaler.scale_[target_indices]

    Y_orig  = Y_scaled  * target_scale + target_mean
    YH_orig = YH_scaled * target_scale + target_mean

    print("\n=== 평가 지표 (원단위 복원 후) ===")
    for i, name in enumerate(TARGET_COLS):
        mse = mean_squared_error(Y_orig[:, i], YH_orig[:, i])
        mae = mean_absolute_error(Y_orig[:, i], YH_orig[:, i])
        r2  = r2_score(Y_orig[:, i], YH_orig[:, i])
        print(f"[{name}]  MSE={mse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    return Y_orig, YH_orig


def plot_results(epoch_losses, test_epoch_losses, Y_orig, YH_orig,
                 results_dir: str, skip_epochs: int = 5):
    """학습 곡선 및 예측 결과 시각화를 results/ 에 저장한다."""
    # Loss curve
    x_axis = range(skip_epochs, len(epoch_losses) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, epoch_losses[skip_epochs-1:], label='Train Loss')
    plt.plot(x_axis, test_epoch_losses[skip_epochs-1:], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted MSE Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'), dpi=150)
    plt.close()
    print(f"[Plot] Loss curve 저장 완료")

    # Prediction vs True
    for i, name in enumerate(TARGET_COLS):
        plt.figure(figsize=(12, 4))
        plt.plot(Y_orig[:200, i], label='True', linewidth=1.5)
        plt.plot(YH_orig[:200, i], label='Pred', linewidth=1.2, linestyle='--')
        plt.xlabel('Sample index')
        plt.ylabel(name)
        plt.title(f'True vs Predicted — {name}')
        plt.legend()
        plt.tight_layout()
        fname = name.replace('%', 'pct').replace(' ', '_') + '.png'
        plt.savefig(os.path.join(results_dir, fname), dpi=150)
        plt.close()
    print(f"[Plot] 예측 결과 그래프 저장 완료")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='data/MiningProcess_Flotation_Plant_Database.csv')
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # 1. 전처리
    print("=" * 50)
    print("[Step 1] 전처리")
    print("=" * 50)
    train_df, test_df, scaler = run_preprocessing(
        args.data_path,
        seq_len=HPARAMS['seq_len'],
        train_ratio=HPARAMS['train_ratio']
    )

    # 2. Dataset / DataLoader
    print("\n" + "=" * 50)
    print("[Step 2] Dataset 및 DataLoader 생성")
    print("=" * 50)
    train_loader, test_loader, train_ds = build_loaders(train_df, test_df)

    # 3. 모델 초기화
    print("\n" + "=" * 50)
    print("[Step 3] 모델 초기화")
    print("=" * 50)
    model, device = build_model(train_ds)

    # 4. 학습
    print("\n" + "=" * 50)
    print("[Step 4] 학습")
    print("=" * 50)
    model, epoch_losses, test_epoch_losses = train(
        model, train_loader, test_loader, device, args.results_dir
    )

    # 5. 평가
    print("\n" + "=" * 50)
    print("[Step 5] 평가")
    print("=" * 50)
    Y_orig, YH_orig = evaluate(model, test_loader, device, scaler, args.results_dir)

    # 6. 시각화
    print("\n" + "=" * 50)
    print("[Step 6] 시각화 저장")
    print("=" * 50)
    plot_results(epoch_losses, test_epoch_losses, Y_orig, YH_orig, args.results_dir)

    print("\n모든 과정 완료.")


if __name__ == '__main__':
    main()
