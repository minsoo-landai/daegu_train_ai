# AutoEncoder 학습 + 분석 시각화 + ROC 기반 임계값 최적화 포함

import os
import librosa
import librosa.display
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import io
from tensorflow.keras.callbacks import EarlyStopping
import glob
import subprocess, os, soundfile as sf
import tempfile
from tqdm import tqdm

CLIP_DURATION = 3

def load_audio(path, sr):
    try:
        y, native_sr = sf.read(path)
        if sr and native_sr != sr:
            y = librosa.resample(y.T, orig_sr=native_sr, target_sr=sr)
        return y
    except Exception as e:
        print(f"[!] {path} 읽기 실패 ({e}) → ffmpeg 변환 시도")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        result = subprocess.run([
            "ffmpeg", "-y", "-i", path,
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sr),
            tmp_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result.returncode != 0 or not os.path.exists(tmp_path):
            print(f"[X] ffmpeg 변환 실패: {path} → 파일 건너뜀")
            return None

        y, _ = librosa.load(tmp_path, sr=sr)
        os.remove(tmp_path)
        return y

def split_audio(y, sr):
    clip_len = int(sr * CLIP_DURATION)
    clips = []
    for i in range(0, len(y), clip_len):
        clip = y[i:i+clip_len]
        if len(clip) < clip_len:
            # 너무 짧으면 버리거나 패딩
            continue
        clips.append(clip)
    return clips

def compute_mfcc(y, sr):
    """
    1) n_ffcc : DTW 계산시 사용하는 MFCC 특징 수
    - 영향도 : 너무 적으면 정보 부족, 너무 많으면 민감성 증가
    - 튜닝 팁 : 일반적으로 13~20 사이(기본 13) / 노이즈 환경에서는 축소 가능
    - 각 오디오 클 [13 x T] 형태 행렬로 요약됨 (T : 시간 프레임 수)
    2) MFCC : 사람의 청각 특성에 맞게 오디오 주파수 정보를 요약한 대표적 특징
    - 추출 흐름 :
        오디오 > 프레임 분할 > FFT(주파수 변환) > Mel Scale 필터 적용 > 로그화 > DCT > MRCC 벡터
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = mfcc.T   # (frames, 13)
    # 빈 배열 방지
    if mfcc.shape[0] == 0:  # 프레임 없으면 skip
        return None
    return mfcc


def compute_dtw_distance(mfcc1, mfcc2):
    """
    - D : 누적 비용 행렬 (두 시퀀스 매칭시키는데 드는 비용을 좌상단-우하단까지 계산한 2차원 배열)
    - D[-1,-1] : 전체 매칭 끝났을 때 최종 비용 (최소 누적 거리)
    - 두 전체 시퀀스 간 최종 DTW 거리

    # 항상 (특징 차원, 시간 프레임) >>> (시간 프레임, 특징 차원) 변환
    #### librosa.feature.mfcc 출력은 (n_mfcc, T) 형태 
    #### mfcc = 13 / T = 프레임 수
    #### librosa.sequence.dtw는 두 입력 행 개수가 같아야 (특징 차원 통일)
    """
    if mfcc1 is None or mfcc2 is None:
        return None
    # 두 입력 모두 (frames, features) 구조
    if mfcc1.shape[1] != mfcc2.shape[1]:
        print(f"[SKIP] Feature mismatch: {mfcc1.shape} vs {mfcc2.shape}")
        return None
    try:
        D, _ = librosa.sequence.dtw(X=mfcc1, Y=mfcc2, metric='euclidean')
        return D[-1, -1]
    except Exception as e:
        #print(f"[SKIP] DTW 실패: {e}")
        return None

def audio_to_spectrogram_image(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(S_dB, sr=sr)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = tf.keras.preprocessing.image.load_img(buf, target_size=(224, 224), color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return img_array

def build_autoencoder_model():
    input_img = Input(shape=(224, 224, 1))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    encoded = layers.Conv2D(8, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D(2)(encoded)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    model = models.Model(input_img, decoded)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model

def suggest_sampling_rate(path):
    candidate_srs = [8000, 16000, 22050, 44100]
    scores = []
    for sr in candidate_srs:
        y, _ = librosa.load(path, sr=sr, duration=5.0)
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        total_energy = np.sum(S)
        high_energy = np.sum(S[freqs > (sr / 4)])
        scores.append(high_energy / total_energy)
    best_sr = candidate_srs[np.argmax(scores)]
    print(f"[\u2714] 자동 선택된 SR: {best_sr}Hz")
    return best_sr

def train_autoencoder(normal_paths, reference_path, model_save_path="autoencoder.h5", thresholds_path="thresholds.npz", filter_strict=True):
    import os
    import librosa
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input
    from sklearn.model_selection import train_test_split
    import io

    CLIP_DURATION = 3

    # --- 파일 필터링 ---
    print(f"[INFO] 원본 파일 개수: {len(normal_paths)}")
    filtered_paths = []
    for p in normal_paths:
        y = load_audio(p, 16000)   # SR은 자동 선택 전이지만 대략 16k로 테스트
        if y is not None and len(y) > 1000:  # 최소 길이 체크 (짧은 쓰레기 데이터 제외)
            filtered_paths.append(p)
        else:
            print(f"[SKIP] 손상되었거나 너무 짧아서 제외 → {p}")

    normal_paths = filtered_paths
    print(f"[INFO] 사용 가능한 정상 파일 개수: {len(normal_paths)}")
    print("")


    def suggest_sampling_rate(path, candidate_srs=[8000, 16000, 22050, 44100, 48000]):
        """
        - 적절한 sr 자동 선택해 고주파 특징을 보존하고 모델 입력 품질 향상
        - 대표성 있으면서 계산이 빠르도록 5초간 오디오 불러와 STFT 수행 -> 스펙트럼 S 추출
        """
        high_energy_ratios = []
        for sr in candidate_srs:
            y, _ = librosa.load(path, sr=sr, duration=5.0)        # 5초간 오디오를 sr로 로드
            # 시간에 따른 주파수 분포(=스펙트럼) 계산
            S = np.abs(librosa.stft(y))                           # STFT 수행해 스펙트럼 추출 / abs() 진폭=에너지 근사값 변환
            # 주파수 벡터 생성 (STFT 배열의 각 row가 몇 Hz 의미하는지 매핑)
            freqs = librosa.fft_frequencies(sr=sr)                # 주파수 벡터 생성
            total_energy = np.sum(S)
            high_energy = np.sum(S[freqs > (sr / 4)])             # 고주파수 에너지 비중 계산 (1/4 이상을 고주파로 간주해 해당 대역 에너지 합산)
            high_energy_ratios.append(high_energy / total_energy) # 비율 계산 (전체 에너지 중 고주파 비중 => 이 샘플레이트에서 고주파 정보 얼마나 살리)
        best_sr = candidate_srs[np.argmax(high_energy_ratios)]    # 가장 높은 비율 보이는 SR 선택
        print(f"[✔] 자동 선택된 SR: {best_sr}Hz (고주파 에너지 반영)")
        return best_sr

    def audio_to_spectrogram_image(y, sr):
        # 시간축, 주파수 축(Mel-scaled)  구성된 2D 에너지 분포
        # Mel scale : 인간 청각 특성에 맞춘 주파수 스케일. 고주파 영역 세분화
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        # dB로 스케일 변환 (log 스케일 변환)
        # power -> decibel / ref는 가장 큰 에너지 기준 0dB 설정 (시각적 명확한 명암 대비)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 사이즈 (224,224)
        fig = plt.figure(figsize=(2.24, 2.24), dpi=100) # 2.24 * 100dpi -> 224픽셀
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None) # 좌표축, 여백 제거로 순수 이미지 정보만
        plt.axis('off')

        # 메모리 버퍼로 저장 (디스크 저장x) - 메모리 내에서 PNG 생성
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0) # 가장자리 여백 제거
        plt.close(fig)
        buf.seek(0)

        # 이미지 로딩 및 전처리 (224,224,1) / float32 / 0~1
        img = tf.keras.preprocessing.image.load_img(buf,
                                                    target_size=(224, 224),   # CNN 모델과 일치시킴
                                                    color_mode='grayscale')   # 단일 채널
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0    # (224,224,1) 크기 float32 numpy 배열 변환 / 정규화
        return img_array

    def estimate_dtw_threshold(normal_paths, reference_path, sr, percentile=90):
        """
          * 추론시 각 오디오 클립이 ref 오디오와 얼마나 다른지 판단하는 기준
          - DTW : 두 시계열 데이터간 유사도를 측정하는 알고리즘
            - 속도 차이나 시간축 왜곡 있는 시계열 데이터를 유연하게 정렬해 유사도(거리) 계산
          - 두 가지 방식을 혼합해 Threshold 결정 (임계값 전략) - outlier에 덜 민감, 안정ㅇ적
            1) Percentile 기준 (상위 90%)
            2) 평균 + 표준편차 (Z-score)
        """
        ref_audio = load_audio(reference_path, sr)  # 정상 오디오
        if ref_audio is None:
            raise RuntimeError("[X] 참조 오디오 로드 실패")
            
        ref_clip = split_audio(ref_audio, sr)[0]    # 참조 오디오 첫 3초 클립 > MFCC 특징 벡터
        ref_mfcc = compute_mfcc(ref_clip, sr)
        if ref_mfcc is None:
            raise RuntimeError("[X] 참조 오디오에서 MFCC 추출 실패")

        dtw_scores = []
        
        #for path in normal_paths:
        for path in tqdm(normal_paths, desc="DTW threshold 계산중"):
            y = load_audio(path, sr)
            if y is None:
                continue
            for clip in split_audio(y, sr):         # 3초 단위 분할
                mfcc = compute_mfcc(clip, sr)       # 각 클립을 참조 클립과 DTW 거리 계산
                score = compute_dtw_distance(mfcc, ref_mfcc)    # DTW 거리 저장
                if score is None:
                    #print(f"[SKIP] DTW 실패: {path}")
                    continue
                dtw_scores.append(score)

        percentile_th = np.percentile(dtw_scores, percentile)   # 상위 10% 컷
        # 평균보다 2표준편차 이상 벗어난 경우 이상값 (정규분포시 약 95% 범위) - 극단값 배제
        z_th = np.mean(dtw_scores) + 2.0 * np.std(dtw_scores)   # 평균 + 2*표준편차
        # 두 기준 중 더 보수적인 값 사용 (FN- 이상인데 정상 판단 줄이는 전략)
        threshold = min(percentile_th, z_th)
        print(f"[✔] DTW Threshold 자동 설정: {threshold:.2f} (p{percentile} & z=2 병합)")
        return threshold, dtw_scores

    def build_autoencoder_model():
        input_img = Input(shape=(224, 224, 1))
        # 인코더 (3-2-1)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D(2, padding='same')(x)
        x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2, padding='same')(x)
        encoded = layers.Conv2D(8, 3, activation='relu', padding='same')(x)

        # 디코더 (1-2-3)
        x = layers.UpSampling2D(2)(encoded)
        x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling2D(2)(x)
        decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

        model = models.Model(input_img, decoded)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.MeanSquaredError() # AE Loss # 입력 이미지, 출력 이미지의 MSE 기준 파라미터 업데이트
                      )
        return model

    def prepare_training_data(normal_paths, reference_path, sr, threshold, filter_strict=True):
        """
        - Train Data 필터링 (선택)
        - DTW 거리 < Threshold인 클립만 학습에 사용
        - filter_strict : 학습시 DTW threshold 이상 데이터 제외 여부
          - True : 정제된 정상만 학습
          - False : 노이즈 포함 학습
        """
        X = []
        ref_y = load_audio(reference_path, sr)
        if ref_y is None:
            raise RuntimeError("[X] 참조 오디오 로드 실패")

        ref_mfcc = compute_mfcc(split_audio(ref_y, sr)[0], sr)

        #for path in normal_paths:
        for path in tqdm(normal_paths, desc="학습 데이터 준비중"):
            y = load_audio(path, sr)
            if y is None:
                continue 
            for clip in split_audio(y, sr):
                mfcc = compute_mfcc(clip, sr)
                score = compute_dtw_distance(mfcc, ref_mfcc)
                if not filter_strict or score < threshold:
                    img = audio_to_spectrogram_image(clip, sr)
                    X.append(img)

        return np.array(X)

    # ----- 학습 시작 -----
    sr = suggest_sampling_rate(reference_path)
    threshold, dtw_scores = estimate_dtw_threshold(normal_paths, reference_path, sr)
    X = prepare_training_data(normal_paths, reference_path, sr, threshold, filter_strict=filter_strict)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    print(f"[INFO] 학습 데이터: {X_train.shape[0]} / 검증: {X_val.shape[0]}")
    print(f"                    {round(((X_train.shape[0])/(X_train.shape[0]+X_val.shape[0]))*100,2)}% : {round(((X_val.shape[0])/(X_train.shape[0]+X_val.shape[0]))*100,2)}%")
    print("")
    model = build_autoencoder_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, X_train,
                        epochs=300, #100
                        batch_size=16,
                        validation_data=(X_val, X_val),
                        shuffle=True,
                        callbacks=[early_stopping])
    model.save(model_save_path)
    print(f"[완료] 모델 저장됨 → {model_save_path}")

    # Threshold 저장
    # ae loss = 입력 이미지와 복원 이미지 간 MSE
    ae_losses = [np.mean((x - model.predict(x[np.newaxis, ...], verbose=0)[0])**2) for x in X_val]
    ae_threshold = np.percentile(ae_losses, 90)
    np.savez(thresholds_path, ae_threshold=ae_threshold, dtw_threshold=threshold)
    print(f"[✔] Thresholds 저장됨 → {thresholds_path}")

    # AE Loss 분포 시각화
    plt.figure(figsize=(10, 4))
    plt.hist(ae_losses, bins=30, alpha=0.7, label='AE Loss (val)')
    plt.axvline(ae_threshold, color='r', linestyle='--', label=f'Threshold: {ae_threshold:.5f}')
    plt.title('Autoencoder Reconstruction Loss Distribution')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, sr, threshold

normal_paths  = glob.glob('../deagu_manufacture_ai/data/train_data/sound/*.wav', recursive=True)   # 하위 디렉토리까지 재귀적으로 검색
reference_path = "../deagu_manufacture_ai/data/reference/reference.wav"

model, sr_used, dtw_threshold = train_autoencoder(normal_paths, reference_path)