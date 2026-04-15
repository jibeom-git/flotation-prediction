"""
model.py
--------
FlotationNetWithWavelet: LSTM 시간 branch + 웨이블릿 MLP branch 결합 모델

아키텍처 설계 근거:
- 논문(Pu et al., 2020)의 FlotationNet을 백본으로 사용
- LSTM branch: 4개 주요 시계열(Iron/Silica Feed + Concentrate)의 시간적 의존성 포착
- Wavelet MLP branch: 동일 시계열의 주파수 도메인 특징(db4 분해 계수) 학습
- Auxiliary pass-through: 19개 공정 파라미터를 concat에 직접 포함
  (논문의 "condiment function" 개념 — 별도 변환 없이 그대로 concat)
- 가중 손실: 논문과 동일하게 Fe(λ1=0.3), Si(λ2=0.7) 설정
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletMLP(nn.Module):
    """
    웨이블릿 계수 벡터를 입력받아 압축된 표현으로 변환하는 MLP.

    입력: (B, D_w) — 웨이블릿 분해 계수
    출력: (B, out_dim)
    """

    def __init__(self, w_dim: int, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlotationNetWithWavelet(nn.Module):
    """
    LSTM(시간 도메인) + WaveletMLP(주파수 도메인) 결합 모델.

    입력:
        x_seq  : (B, T, 4)  — 주요 시계열 4채널
        x_aux  : (B, P)     — 공정 파라미터 (pass-through)
        w_feat : (B, D_w)   — 웨이블릿 분해 계수

    출력:
        (B, 2) — [% Iron Concentrate, % Silica Concentrate] 예측값

    Forward 흐름:
        x_seq → LSTM → h_last (B, lstm_hidden)
        w_feat → WaveletMLP → w_out (B, wavelet_out_dim)
        [h_last, w_out, x_aux] → concat → FC1 → FC2 → FC3 → 출력
    """

    def __init__(self,
                 aux_dim: int,
                 w_dim: int,
                 wavelet_out_dim: int = 64,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 2,
                 dense_hidden: int = 64,
                 in_channels: int = 4):
        super().__init__()

        # --- 시간 도메인 branch: Stacked LSTM ---
        # batch_first=True: 입력 shape (B, T, C)로 받음
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0  # 다층 LSTM 과적합 방지
        )

        # --- 주파수 도메인 branch: Wavelet MLP ---
        # w_dim은 외부에서 wavelet feature 차원으로 초기화
        self.wavelet_mlp = WaveletMLP(
            w_dim=w_dim,
            hidden_dim=dense_hidden,
            out_dim=wavelet_out_dim
        )

        # --- 최종 회귀 MLP ---
        # concat_dim = LSTM 출력 + Wavelet 출력 + 공정 파라미터
        concat_dim = lstm_hidden + wavelet_out_dim + aux_dim
        self.fc = nn.Sequential(
            nn.Linear(concat_dim, dense_hidden),
            nn.ReLU(),
            nn.Linear(dense_hidden, dense_hidden),
            nn.ReLU(),
            nn.Linear(dense_hidden, 2),  # [Fe 예측, Si 예측]
        )

    def forward(self, x_seq: torch.Tensor,
                x_aux: torch.Tensor,
                w_feat: torch.Tensor) -> torch.Tensor:
        # 1) LSTM — 마지막 타임스텝의 hidden state만 사용
        lstm_out, _ = self.lstm(x_seq)      # (B, T, lstm_hidden)
        h_last = lstm_out[:, -1, :]         # (B, lstm_hidden)

        # 2) Wavelet branch
        w_out = self.wavelet_mlp(w_feat)    # (B, wavelet_out_dim)

        # 3) concat → 최종 FC
        fused = torch.cat([h_last, w_out, x_aux], dim=1)
        return self.fc(fused)               # (B, 2)


def weighted_mse(y_pred: torch.Tensor, y_true: torch.Tensor,
                 lam1: float = 0.3, lam2: float = 0.7) -> torch.Tensor:
    """
    논문(Pu et al., 2020) 식 (3)과 동일한 가중 MSE 손실.

    L = λ1 * MSE(Fe_pred, Fe_true) + λ2 * MSE(Si_pred, Si_true)

    가중치 설정 근거:
    - 부유선광의 주 목적은 Fe 정제 → Fe 예측 정확도 우선
    - λ2 > λ1으로 Si 오차에 더 큰 패널티를 부여하여
      Si 예측 손실이 전체 손실의 70%를 차지하도록 유도
    - StandardScaler 적용 후이므로 Fe(~65%)와 Si(~2%)의 절대값 차이는 제거됨
    """
    loss_fe = F.mse_loss(y_pred[:, 0], y_true[:, 0])
    loss_si = F.mse_loss(y_pred[:, 1], y_true[:, 1])
    return lam1 * loss_fe + lam2 * loss_si
