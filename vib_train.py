import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import glob

#### GPU 하나로는 224 * 224 돌리기 어려움. OOM 
#### CUDA_VISIBLE_DEVICES="" python vib_train.py

def load_and_preprocess_vibration_data_from_folder(folder_path, fs=16, duration=3):
    """
    폴더 내의 모든 CSV 파일을 로드하고 각 파일을 개별적으로 처리합니다.
    각 파일이 이미 3초 데이터라면 파일별로 하나의 윈도우로 처리합니다.

    Args:
        folder_path (str): CSV 파일들이 있는 폴더 경로
        fs (int): 샘플링 주파수 (Hz) - 기본값을 16Hz로 변경
        duration (int): 윈도우 길이 (초)

    Returns:
        dict: {"Acc1": [...], "Acc2": [...], "Acc3": [...], "filenames": [...]}
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    csv_files.sort()  # 파일명 순서대로 정렬
    
    print(f"▶ {len(csv_files)}개의 CSV 파일을 찾았습니다.")
    
    all_acc1, all_acc2, all_acc3 = [], [], []
    all_filenames = []  # 파일명 추적용
    
    for csv_file in tqdm(csv_files, desc="CSV 파일 로딩 중"):
        try:
            # CSV 파일 로드 (헤더가 있는 경우)
            df = pd.read_csv(csv_file)
            
            # 컬럼명 확인 및 매핑
            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                # 새로운 형식: x, y, z 컬럼
                acc1 = df['x'].values.astype(float)
                acc2 = df['y'].values.astype(float)
                acc3 = df['z'].values.astype(float)
            elif 'Acc1' in df.columns and 'Acc2' in df.columns and 'Acc3' in df.columns:
                # 기존 형식: Acc1, Acc2, Acc3 컬럼
                acc1 = df['Acc1'].values.astype(float)
                acc2 = df['Acc2'].values.astype(float)
                acc3 = df['Acc3'].values.astype(float)
            else:
                print(f"⚠ 경고: {csv_file}에서 적절한 컬럼을 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # 파일별로 개별 처리
            filename = os.path.basename(csv_file)
            
            # 각 파일을 하나의 윈도우로 처리
            all_acc1.append(acc1)
            all_acc2.append(acc2)
            all_acc3.append(acc3)
            all_filenames.append(filename)
            
        except Exception as e:
            print(f"⚠ 오류: {csv_file} 로딩 중 오류 발생 - {e}")
            continue
    
    if not all_acc1:
        raise ValueError("로드된 데이터가 없습니다. 폴더 경로와 CSV 파일 형식을 확인해주세요.")
    
    print(f"▶ 총 {len(all_acc1)}개 파일 로드 완료")
    
    # 각 파일의 데이터 길이 확인
    file_lengths = [len(acc1) for acc1 in all_acc1]
    print(f"▶ 파일별 데이터 길이: 최소 {min(file_lengths)}, 최대 {max(file_lengths)}")
    
    # 윈도우 크기 확인 (실제 데이터에 맞게 조정)
    expected_win_len = fs * duration  # 16Hz × 3초 = 48개 샘플
    actual_min_len = min(file_lengths)
    
    # 실제 데이터 길이에 맞게 윈도우 크기 조정
    if actual_min_len < expected_win_len:
        print(f"⚠ 실제 데이터 길이({actual_min_len})가 예상 길이({expected_win_len})보다 짧습니다.")
        print(f"⚠ 윈도우 크기를 {actual_min_len}개 샘플로 조정합니다.")
        win_len = actual_min_len
    else:
        win_len = expected_win_len
    
    print(f"▶ 최종 윈도우 크기: {win_len}개 샘플")
    
    # 데이터 길이가 윈도우 크기와 다른 경우 처리
    processed_acc1, processed_acc2, processed_acc3, processed_filenames = [], [], [], []
    
    for i, (acc1, acc2, acc3, filename) in enumerate(zip(all_acc1, all_acc2, all_acc3, all_filenames)):
        if len(acc1) >= win_len:
            # 윈도우 크기만큼만 사용
            processed_acc1.append(acc1[:win_len])
            processed_acc2.append(acc2[:win_len])
            processed_acc3.append(acc3[:win_len])
            processed_filenames.append(filename)
        else:
            # 윈도우 크기보다 작으면 패딩 또는 건너뛰기
            print(f"⚠ 경고: {filename}의 데이터가 너무 짧습니다 ({len(acc1)} < {win_len}). 건너뜁니다.")
    
    vib_xyz_dict = {
        "Acc1": processed_acc1,
        "Acc2": processed_acc2,
        "Acc3": processed_acc3,
        "filenames": processed_filenames,
    }

    print(f"▶ {len(vib_xyz_dict['Acc1'])}개의 윈도우 생성 완료")
    return vib_xyz_dict

def vibration_to_spectrogram_array(signal, fs=16):
    """
    #### 1D 신호 >>> 2D 이미지(Time-Frequency Spectrogram) 변환
    : 시간-주파수 영역으로 변환 위해 STFT(Short-Time Fouier Transform) 기반 스펙트트럼 사용

    1) 파라미터 설정값 - 데이터 길이에 맞게 동적 조정
    - fs : 실제 샘플링 주파수 (16Hz)
    - nperseg : 데이터 길이에 맞게 조정 (최소 4개 샘플)
    - noverlap : nperseg의 절반 이하로 설정

    ** 제조 설비 진동은 보통 10ms ~ 수백 ms 사이 순간적 이상 이벤트 발생
      0.25초 단위는 한 주기 또는 이상 발생 패턴 하나를 포착하기에 충분한 시간 분해능 가짐
    * 50% overlap(추가 고민 필요) : 고주파 불량 신호가 짧은시간 동안 고에너지를 띌 경우 겹침 처리로 누락 방지
    """
    # 데이터 길이에 맞게 파라미터 동적 조정
    signal_length = len(signal)
    
    # nperseg를 데이터 길이의 절반으로 설정 (최소 4개)
    nperseg = max(4, min(signal_length // 2, 256))
    
    # noverlap을 nperseg의 절반으로 설정 (nperseg보다 작게)
    noverlap = max(1, nperseg // 2)
    
    # 데이터가 너무 짧으면 전체 데이터를 사용
    if signal_length < nperseg:
        nperseg = signal_length
        noverlap = 0
    
    print(f"   스펙트로그램 파라미터: nperseg={nperseg}, noverlap={noverlap}, signal_length={signal_length}")
    
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # 중심 주파수 계산 (weighted mean)
    center_freq = np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-8)

    # 대역폭 계산 (variance) : 신호가 퍼진 정도 (주파수 분산)
    #### 정상 신호는 좁은 대역, 이상 신호는 넓은 대역에 퍼짐
    bandwidth = np.sqrt(np.sum((f[:, None] - center_freq)**2 * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-8))

    # 로그 변환 → 고주파 강조 (dB 단위로 변환)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)

    # 정규화 (전체 기준 or 고정 min/max)
    Sxx_log = (Sxx_log - np.min(Sxx_log)) / (np.max(Sxx_log) - np.min(Sxx_log))

    # (선택) scalogram ridge 검출 및 비교용
    # ridge_curve = np.argmax(Sxx_log, axis=0)

    Sxx_resized = tf.image.resize(Sxx_log[..., np.newaxis], (224, 224)).numpy()

    return Sxx_resized.astype(np.float32)

def generate_images_and_save(vib_xyz_dict, save_path="vib_images.npy", fs=1000):
    """
    진동 데이터를 스펙트로그램 이미지로 변환하여 저장합니다.
    """
    total = len(vib_xyz_dict["Acc1"])
    npy_writer = np.lib.format.open_memmap(save_path, mode='w+', dtype=np.float32, shape=(total, 224, 224, 1))
    
    for i in tqdm(range(total), desc="스펙트로그램 생성 중", ncols=100):
        merged = np.mean([vib_xyz_dict["Acc1"][i], vib_xyz_dict["Acc2"][i], vib_xyz_dict["Acc3"][i]], axis=0)
        npy_writer[i] = vibration_to_spectrogram_array(merged, fs)
        if i % 100 == 0: 
            gc.collect()
    
    del npy_writer
    gc.collect()
    print(f"▶ 스펙트로그램 이미지 저장 완료: {save_path}")

def build_cnn_autoencoder():
    """
    CNN AutoEncoder 모델을 생성합니다.
    
    1. 구조
      1) 이미지 크기 : (224 * 224 * 1)
      2) Encoder : Conv(16) > Pool > Conv(8) > upsample > Conv(4)
      3) Decoder : upsample > Conv(8) > upsample > Conv(1, sigmoid)
      4) Activation : ReLU / 출력 Sigmoid
      5) Loss : MSE
    """
    input_img = Input(shape=(224, 224, 1))
    
    # Encoder
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(8, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    encoded = layers.Conv2D(4, 3, activation='relu', padding='same')(x)
    
    # Decoder
    x = layers.UpSampling2D(2)(encoded)
    x = layers.Conv2D(8, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    
    model = models.Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mse')
    
    print("▶ CNN AutoEncoder 모델 생성 완료")
    model.summary()
    
    return model

def compute_2d_dtw(img1, img2):
    """
    두 개의 2D 이미지 (스펙트로그램)를 시간 축 기준 주파수 벡터 시퀀스로 보고 DTW 거리 계산
    img1, img2 shape: (freq, time)
    - 시간 벡터 : 주파수 스펙트럼
    - 앞선 log 변환 통해 고주파 잔진동 충분히 반영. (DTW 신호의 전체적 모양을 제대로 인식하게)

    * 1D 아닌 2D 스펙트로그램 기반 시간별 주파수 벡터 기반 시계열 매칭
    * 진동 신호 이상은 주로 특정 주파수 대역 에너지 변화로 발생.
      전체 waveform 모양보다는 시간별 주파수 스펙트럼의 변화가 더 유의미.
    * AE는 공간적 손실만 평가함. 시간축 왜곡이나 진동 패턴 왜곡까지 평가하기 위해 DTW 보완
    """
    seq1 = [img1[:, i] for i in range(img1.shape[1])]  # 시간축 따라 자르기
    seq2 = [img2[:, i] for i in range(img2.shape[1])]
    dist, _ = fastdtw(seq1, seq2, dist=euclidean)  # DTW with vector distance
    return dist

def train_vibration_cnn_ae_dtw(
    train_folder_path,
    model_path="vib_cnn_ae.keras",
    threshold_path="vib_cnn_thresh.npy",
    cache_path="vib_images.npy",
    fs=1000
):
    """
    진동 이상 탐지를 위한 CNN AutoEncoder + DTW 모델을 학습합니다.
    
    Args:
        train_folder_path (str): 학습 데이터 폴더 경로
        model_path (str): 모델 저장 경로
        threshold_path (str): 임계값 저장 경로
        cache_path (str): 스펙트로그램 이미지 캐시 경로
        fs (int): 샘플링 주파수
    
    Returns:
        tuple: (학습된 모델, 임계값)
    """
    print("=" * 60)
    print("진동 이상 탐지 모델 학습 시작")
    print("=" * 60)
    
    # 1. 폴더에서 데이터 로드
    print(f"\n1단계: 학습 데이터 로딩")
    print(f"   경로: {train_folder_path}")
    vib_xyz_dict = load_and_preprocess_vibration_data_from_folder(train_folder_path, fs=fs)
    
    # 2. 이미지 캐시 생성
    print(f"\n 2단계: 스펙트로그램 이미지 생성")
    if not os.path.exists(cache_path):
        generate_images_and_save(vib_xyz_dict, cache_path, fs)
    else:
        print(f"   캐시 파일이 이미 존재합니다: {cache_path}")

    all_images = np.load(cache_path, mmap_mode='r')
    print(f"   총 {len(all_images)}개 이미지 로드 완료")

    # 3. 데이터 분할
    print(f"\n3단계: 데이터 분할")
    X_train, X_val, idx_train, idx_val = train_test_split(
        all_images, np.arange(len(all_images)), test_size=0.2, random_state=42
    )
    print(f"   학습 데이터: {len(X_train)}개")
    print(f"   검증 데이터: {len(X_val)}개")

    # 4. 모델 정의 및 학습
    print(f"\n4단계: 모델 학습")
    model = build_cnn_autoencoder()
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("학습 시작...")
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=300,
        batch_size=2,
        shuffle=True,
        verbose=1,
        callbacks=[early_stop]
    )

    # 5. 모델 저장
    print(f"\n5단계: 모델 저장")
    model.save(model_path)
    print(f"   모델 저장 완료: {model_path}")

    # 6. 복원 예측
    print(f"\n6단계: 복원 예측 및 임계값 계산")
    print("   검증 데이터로 복원 예측 중...")
    recon_val = model.predict(X_val, batch_size=8, verbose=1)

    # 7. AE Loss 계산
    ae_losses = np.mean((X_val - recon_val) ** 2, axis=(1, 2, 3))
    print(f"   AE Loss 계산 완료: 평균 {np.mean(ae_losses):.6f}")

    # 8. DTW 거리 계산
    print("   DTW 거리 계산 중...")
    dtw_distances = []
    for i, idx in enumerate(tqdm(idx_val, desc="DTW 계산", ncols=50)):
        # 원본 merged 시그널
        merged = np.mean([
            vib_xyz_dict["Acc1"][idx],
            vib_xyz_dict["Acc2"][idx],
            vib_xyz_dict["Acc3"][idx]
        ], axis=0)

        # 원본 시그널 → 스펙트로그램 이미지
        original_img = vibration_to_spectrogram_array(merged, fs).squeeze()  # shape: (224, 224)
        recon_img = recon_val[i].squeeze()  # shape: (224, 224)

        # 2D DTW 계산
        dtw_score = compute_2d_dtw(original_img, recon_img)
        dtw_distances.append(dtw_score)

    # 9. 정규화 및 threshold 계산
    print("   임계값 계산 중...")
    ae_norm = (ae_losses - np.min(ae_losses)) / (np.max(ae_losses) - np.min(ae_losses) + 1e-8)
    dtw_norm = (np.array(dtw_distances) - np.min(dtw_distances)) / (np.max(dtw_distances) - np.min(dtw_distances) + 1e-8)

    # Final Score 계산 (AE Loss + DTW score)
    final_scores = 0.5 * ae_norm + 0.5 * dtw_norm
    threshold = np.percentile(final_scores, 90)
    
    np.save(threshold_path, threshold)
    print(f"   임계값 저장 완료: {threshold_path} (90% 기준 = {threshold:.6f})")

    # 10. 학습 결과 요약
    print(f"\n학습 결과 요약")
    print(f"   최종 검증 손실: {min(history.history['val_loss']):.6f}")
    print(f"   학습 에포크: {len(history.history['val_loss'])}")
    print(f"   임계값 (90%): {threshold:.6f}")
    
    print(f"\n모델 학습 완료!")
    print(f"   모델 파일: {model_path}")
    print(f"   임계값 파일: {threshold_path}")
    
    return model, threshold

def setup_gpu_memory():
    """GPU 메모리 사용량을 제한합니다."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # GPU 메모리 증가를 허용하되 제한
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # GPU 메모리 제한 (예: 4GB)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )
            print("GPU 메모리 제한 설정 완료 (4GB)")
        except RuntimeError as e:
            print(f"GPU 메모리 설정 오류: {e}")

def main():
    """
    메인 실행 함수
    """
    # 설정
    TRAIN_FOLDER = "/home/minsoo0807/deagu_manufacture_ai/data/train_data/vib"
    MODEL_PATH = "vib_cnn_ae.keras"
    THRESHOLD_PATH = "vib_cnn_thresh.npy"
    CACHE_PATH = "vib_images.npy"
    
    # 실제 샘플링 주파수로 수정 (47개 샘플 ÷ 3초 ≈ 16Hz)
    ACTUAL_FS = 16
    
    print("진동 이상 탐지 모델 학습 프로그램")
    print("=" * 50)
    print(f"샘플링 주파수: {ACTUAL_FS}Hz")
    
    # 학습 실행
    try:
        model, threshold = train_vibration_cnn_ae_dtw(
            train_folder_path=TRAIN_FOLDER,
            model_path=MODEL_PATH,
            threshold_path=THRESHOLD_PATH,
            cache_path=CACHE_PATH,
            fs=ACTUAL_FS  # 실제 샘플링 주파수 사용
        )
        
        print(f"\n학습이 성공적으로 완료되었습니다!")
        print(f"   모델 파일: {MODEL_PATH}")
        print(f"   임계값 파일: {THRESHOLD_PATH}")
        
    except Exception as e:
        print(f"\n학습 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()