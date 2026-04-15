"""
dataset.py
----------
FlotationWaveletDataset: 웨이블릿 특징을 사전 계산하여 캐싱하는 Dataset 클래스

설계 근거:
- pywt.wavedec은 numpy 연산이므로 __getitem__ 내부에서 호출하면
  DataLoader의 매 배치마다 반복 계산이 발생하여 학습 병목이 생김
- __init__ 시점에 전체 윈도우의 웨이블릿 특징을 미리 계산하여
  텐서로 캐싱해두는 방식으로 I/O 병목을 제거함
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pywt


def compute_wavelet_feature(x_seq: np.ndarray,
                            wavelet: str = 'db4',
                            level: int = 2) -> np.ndarray:
    """
    단일 시계열 윈도우에 대해 웨이블릿 분해 계수를 추출한다.

    매개변수:
        x_seq   : (T, C) 형태의 numpy 배열
        wavelet : PyWavelets 웨이블릿 종류 (기본 db4 — Daubechies 4)
        level   : 분해 레벨 수

    반환:
        (D_w,) 형태의 1D 배열 — 모든 채널의 계수를 flatten하여 concat

    웨이블릿 선택 근거:
        db4는 시계열 신호의 국소적 특징(edge, spike) 포착에 적합하며
        level=2 분해 시 seq_len=32에서 근사(approximation) + 2개 detail 계수를 생성함
    """
    T, C = x_seq.shape
    feats = []
    for c in range(C):
        coeffs = pywt.wavedec(x_seq[:, c], wavelet=wavelet, level=level)
        feats.append(np.concatenate(coeffs))
    return np.concatenate(feats).astype(np.float32)


class FlotationWaveletDataset(Dataset):
    """
    슬라이딩 윈도우 기반 Dataset.

    입력:
        x_seq  : (T, 4)  — Iron/Silica Feed + Iron/Silica Concentrate 4개 시계열
        x_aux  : (P,)    — 공정 파라미터 19개 (마지막 타임스텝 값)
        w_feat : (D_w,)  — 웨이블릿 분해 계수 (사전 계산, 캐싱)
    출력:
        y      : (2,)    — Iron Concentrate, Silica Concentrate 예측 타겟

    타겟 인덱스 설계:
        x_seq 윈도우의 마지막 타임스텝(end-1)의 타겟을 사용함.
        데이터셋 설명에 따르면 출력값은 이미 2시간 지연이 반영된 상태로
        정렬되어 있으므로, 별도의 shift 없이 동일 타임스텝 사용이 적절함.
    """

    def __init__(self, df, seq_len: int, main_cols: list,
                 proc_param_cols: list, target_cols: list,
                 wavelet: str = 'db4', level: int = 2):
        self.seq_len = seq_len
        df = df.reset_index(drop=True)

        X_main = df[main_cols].to_numpy(dtype=np.float32)
        X_aux  = df[proc_param_cols].to_numpy(dtype=np.float32)
        Y      = df[target_cols].to_numpy(dtype=np.float32)
        N      = len(df)
        M      = N - seq_len + 1  # 생성 가능한 윈도우 수

        # --- 웨이블릿 특징 사전 계산 및 캐싱 ---
        # 매 __getitem__ 호출 시마다 pywt 연산을 반복하지 않도록
        # __init__에서 전체 M개의 윈도우에 대해 미리 계산
        print(f"  웨이블릿 특징 사전 계산 중 (윈도우 수={M})...")
        W_list = []
        for i in range(M):
            W_list.append(compute_wavelet_feature(X_main[i:i+seq_len], wavelet, level))

        # 모든 데이터를 텐서로 변환하여 메모리에 캐싱
        self.X_seq  = torch.from_numpy(
            np.stack([X_main[i:i+seq_len] for i in range(M)])
        )                                              # (M, T, 4)
        self.X_aux  = torch.from_numpy(X_aux[seq_len-1:])   # (M, P)
        self.W_feat = torch.from_numpy(np.stack(W_list))    # (M, D_w)
        self.Y      = torch.from_numpy(Y[seq_len-1:])        # (M, 2)

        print(f"  완료: X_seq={tuple(self.X_seq.shape)}, "
              f"W_feat={tuple(self.W_feat.shape)}, Y={tuple(self.Y.shape)}")

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_aux[idx], self.W_feat[idx], self.Y[idx]
