import wave as wave
import numpy as np
import scipy.signal as sp
import scipy as scipy
import matplotlib.pyplot as plt
import pyaudio

# 順列計算に使用
import itertools
import time


# コントラスト関数の微分（球対称多次元ラプラス分布を仮定）
# s_hat: 分離信号(M, Nk, Lt)
def phi_multivariate_laplacian(s_hat):
    power = np.square(np.abs(s_hat))
    norm = np.sqrt(np.sum(power, axis=1, keepdims=True))

    phi = s_hat / np.maximum(norm, 1.e-18)
    return (phi)


# コントラスト関数の微分（球対称ラプラス分布を仮定）
# s_hat: 分離信号(M, Nk, Lt)
def phi_laplacian(s_hat):
    norm = np.abs(s_hat)
    phi = s_hat / np.maximum(norm, 1.e-18)
    return (phi)


# コントラスト関数（球対称ラプラス分布を仮定）
# s_hat: 分離信号(M, Nk, Lt)
def contrast_laplacian(s_hat):
    norm = 2. * np.abs(s_hat)

    return (norm)


# コントラスト関数（球対称多次元ラプラス分布を仮定）
# s_hat: 分離信号(M, Nk, Lt)
def contrast_multivariate_laplacian(s_hat):
    power = np.square(np.abs(s_hat))
    norm = 2. * np.sqrt(np.sum(power, axis=1, keepdims=True))

    return (norm)


# ICAによる分離フィルタ更新
# x:入力信号( M, Nk, Lt)
# W: 分離フィルタ(Nk,M,M)
# mu: 更新係数
# n_ica_iterations: 繰り返しステップ数
# phi_func: コントラスト関数の微分を与える関数
# contrast_func: コントラスト関数
# is_use_non_holonomic: True (非ホロノミック拘束を用いる） False (用いない）
# return W 分離フィルタ(Nk,M,M) s_hat 出力信号(M,Nk, Lt),cost_buff ICAのコスト (T)
def execute_natural_gradient_ica(x, W, phi_func=phi_laplacian, contrast_func=contrast_laplacian, mu=1.0,
                                 n_ica_iterations=20, is_use_non_holonomic=True):
    # マイクロホン数を取得する
    M = np.shape(x)[0]

    cost_buff = []
    for t in range(n_ica_iterations):
        # 音源分離信号を得る
        s_hat = np.einsum('kmn,nkt->mkt', W, x)

        # コントラスト関数を計算
        G = contrast_func(s_hat)

        # コスト計算
        cost = np.sum(np.mean(G, axis=-1)) - np.sum(2. * np.log(np.abs(np.linalg.det(W))))
        cost_buff.append(cost)

        # コンストラクト関数の微分を取得
        phi = phi_func(s_hat)

        phi_s = np.einsum('mkt,nkt->ktmn', phi, np.conjugate(s_hat))
        phi_s = np.mean(phi_s, axis=1)

        I = np.eye(M, M)
        if is_use_non_holonomic == False:
            deltaW = np.einsum('kmi,kin->kmn', I[None, ...] - phi_s, W)
        else:
            mask = (np.ones((M, M)) - I)[None, ...]
            deltaW = np.einsum('kmi,kin->kmn', np.multiply(mask, -phi_s), W)

        # フィルタを更新する
        W = W + mu * deltaW

    # 最後に出力信号を分離
    s_hat = np.einsum('kmn,nkt->mkt', W, x)

    return (W, s_hat, cost_buff)


# IP法による分離フィルタ更新
# x:入力信号( M, Nk, Lt)
# W: 分離フィルタ(Nk,M,M)
# n_iterations: 繰り返しステップ数
# return W 分離フィルタ(Nk,M,M) s_hat 出力信号(M,Nk, Lt),cost_buff コスト (T)
def execute_ip_multivariate_laplacian_iva(x, W, n_iterations=20):
    # マイクロホン数を取得する
    M = np.shape(x)[0]

    cost_buff = []
    for t in range(n_iterations):

        # 音源分離信号を得る
        s_hat = np.einsum('kmn,nkt->mkt', W, x)

        # 補助変数を更新する
        v = np.sqrt(np.sum(np.square(np.abs(s_hat)), axis=1))

        # コントラスト関数を計算
        G = contrast_multivariate_laplacian(s_hat)

        # コスト計算
        cost = np.sum(np.mean(G, axis=-1)) - np.sum(2. * np.log(np.abs(np.linalg.det(W))))
        cost_buff.append(cost)

        # IP法による更新
        Q = np.einsum('st,mkt,nkt->tksmn', 1. / np.maximum(v, 1.e-18), x, np.conjugate(x))
        Q = np.average(Q, axis=0)

        for source_index in range(M):
            WQ = np.einsum('kmi,kin->kmn', W, Q[:, source_index, :, :])
            invWQ = np.linalg.pinv(WQ)
            W[:, source_index, :] = np.conjugate(invWQ[:, :, source_index])
            wVw = np.einsum('km,kmn,kn->k', W[:, source_index, :], Q[:, source_index, :, :],
                            np.conjugate(W[:, source_index, :]))
            wVw = np.sqrt(np.abs(wVw))
            W[:, source_index, :] = W[:, source_index, :] / np.maximum(wVw[:, None], 1.e-18)

    s_hat = np.einsum('kmn,nkt->mkt', W, x)

    return (W, s_hat, cost_buff)


# 周波数間の振幅相関に基づくパーミュテーション解法
# s_hat: M,Nk,Lt
# return permutation_index_result：周波数毎のパーミュテーション解
def solver_inter_frequency_permutation(s_hat):
    n_sources = np.shape(s_hat)[0]
    n_freqs = np.shape(s_hat)[1]
    n_frames = np.shape(s_hat)[2]

    s_hat_abs = np.abs(s_hat)

    norm_amp = np.sqrt(np.sum(np.square(s_hat_abs), axis=0, keepdims=True))
    s_hat_abs = s_hat_abs / np.maximum(norm_amp, 1.e-18)

    spectral_similarity = np.einsum('mkt,nkt->k', s_hat_abs, s_hat_abs)

    frequency_order = np.argsort(spectral_similarity)

    # 音源間の相関が最も低い周波数からパーミュテーションを解く
    is_first = True
    permutations = list(itertools.permutations(range(n_sources)))
    permutation_index_result = {}

    for freq in frequency_order:

        if is_first == True:
            is_first = False

            # 初期値を設定する
            accumurate_s_abs = s_hat_abs[:, frequency_order[0], :]
            permutation_index_result[freq] = range(n_sources)
        else:
            max_correlation = 0
            max_correlation_perm = None
            for perm in permutations:
                s_hat_abs_temp = s_hat_abs[list(perm), freq, :]
                correlation = np.sum(accumurate_s_abs * s_hat_abs_temp)

                if max_correlation_perm is None:
                    max_correlation_perm = list(perm)
                    max_correlation = correlation
                elif max_correlation < correlation:
                    max_correlation = correlation
                    max_correlation_perm = list(perm)
            permutation_index_result[freq] = max_correlation_perm
            accumurate_s_abs += s_hat_abs[max_correlation_perm, freq, :]

    return (permutation_index_result)


# プロジェクションバックで最終的な出力信号を求める
# s_hat: M,Nk,Lt
# W: 分離フィルタ(Nk,M,M)
# retunr c_hat: マイクロホン位置での分離結果(M,M,Nk,Lt)
def projection_back(s_hat, W):
    # ステアリングベクトルを推定
    A = np.linalg.pinv(W)
    c_hat = np.einsum('kmi,ikt->mikt', A, s_hat)
    return (c_hat)

#file_name: wavファイル
#spectrogram_name: スペクトログラムのファイル名
def make_Spectrogram(file_name, spectrogram_name):
    wav = wave.open(file_name)
    data = wav.readframes(wav.getnframes())
    data = np.frombuffer(data, dtype=np.int16)

    fig = plt.figure(figsize=(10, 4))
    spectrum, freqs, t, im = plt.specgram(data, NFFT=512, noverlap=512 / 16 * 15, Fs=wav.getframerate(), cmap="gray")
    fig.colorbar(im).set_label('Intensity [dB]')
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.savefig("./{}.png".format(spectrogram_name))
    plt.show()
    wav.close()

#file_name: wavファイル
#spectrogram_name: 波形のファイル名
def make_Waveform(file_name, spectrogram_name):
    wav = wave.open(file_name)
    data = wav.readframes(wav.getnframes())
    data = np.frombuffer(data, dtype=np.int16)

    x = np.array(range(wav.getnframes())) / wav.getframerate()
    plt.figure(figsize=(10, 4))
    plt.xlabel("Time [sec]")
    plt.ylabel("Value")
    plt.ylim(-25000, 25000)
    plt.plot(x, data)
    plt.savefig("./{}.png".format(spectrogram_name))
    plt.show()
    wav.close()

# 2バイトに変換してファイルに保存
# signal: time-domain 1d array (float)
# file_name: 出力先のファイル名
# sample_rate: サンプリングレート
def write_file_from_time_signal(signal, file_name, sample_rate):
    # 2バイトのデータに変換
    signal = signal.astype(np.int16)
    # waveファイルに書き込む
    wave_out = wave.open(file_name, 'w')
    # モノラル:1、ステレオ:2
    wave_out.setnchannels(1)
    # サンプルサイズ2byte
    wave_out.setsampwidth(2)
    # サンプリング周波数
    wave_out.setframerate(sample_rate)
    # データを書き込み
    wave_out.writeframes(signal)
    # ファイルを閉じる
    wave_out.close()

#wavファイルの読み込み
def file_read(sounds):
    # 長さを調べる
    n_samples = 0
    # ファイルを読み込む
    for clean_wave_file in sounds:
        wav = wave.open(clean_wave_file)
        if n_samples < wav.getnframes():
            n_samples = wav.getnframes()
        wav.close()

    # 畳み込んだ波形を取得する(チャンネル、サンプル）
    clean_data = np.zeros([len(sounds), n_samples])

    # ファイルを読み込む
    s = 0
    for clean_wave_file in sounds:
        wav = wave.open(clean_wave_file)
        data = wav.readframes(wav.getnframes())
        data = np.frombuffer(data, dtype=np.int16)
        data = data / np.iinfo(np.int16).max
        clean_data[s, :wav.getnframes()] = data
        wav.close()
        s = s + 1

    return (clean_data)


def PlayWavFie(Filename):
    try:
        wf = wave.open(Filename, "r")
    except FileNotFoundError:  # ファイルが存在しなかった場合
        print("[Error 404] No such file or directory: " + Filename)
        return 0

    # ストリームを開く
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 音声を再生
    chunk = 1024
    data = wf.readframes(chunk)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.close()
    print("1")
    p.terminate()
    print("2")



# 乱数の種を初期化
np.random.seed(0)
# サンプリング周波数
sample_rate = 16000

#録音の処理
RESPEAKER_CHANNELS = 4
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = 2
CHUNK = 1024
RECORD_SECONDS = int(input("録音時間を入力"))
WAVE_OUTPUT_FILENAME_1 = "rerere-02.wav"
WAVE_OUTPUT_FILENAME_2 = "rerere-03.wav"

p = pyaudio.PyAudio()

stream = p.open(
            rate=sample_rate,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=RESPEAKER_INDEX,)

print("* recording")

frames_1 = []
frames_2 = [] 

for i in range(0, int(sample_rate / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    a = np.frombuffer(data,dtype=np.int16)[0::4]
    b = np.frombuffer(data,dtype=np.int16)[1::4]
    
    frames_1.append(a.tostring())
    frames_2.append(b.tostring())
    

print("* finish")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME_1, 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(sample_rate)
wf.writeframes(b''.join(frames_1))
wf.close()

wf = wave.open(WAVE_OUTPUT_FILENAME_2, 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(sample_rate)
wf.writeframes(b''.join(frames_2))
wf.close()


#音源分離の処理
wave_name = "rerere"

# 各マイクロホンの入力信号
clean_wave_files = ["./{}-02.wav".format(wave_name),
                    "./{}-03.wav".format(wave_name)]

# マイクロホン数len(clean_wave_files)
n_sources = len(clean_wave_files)

multi_conv_data = file_read(clean_wave_files)


make_Spectrogram("./{}-02.wav".format(wave_name), "./spectrogram_{}-02".format(wave_name))
make_Waveform("./{}-02.wav".format(wave_name), "wave_{}-02".format(wave_name))

# フレームサイズ
N = 1024
# 周波数の数
Nk = int(N / 2 + 1)

# 各ビンの周波数
freqs = np.arange(0, Nk, 1) * sample_rate / N

# 短時間フーリエ変換を行う
f, t, stft_data = sp.stft(multi_conv_data, fs=sample_rate, window="hann", nperseg=N)

# ICAの繰り返し回数
n_ica_iterations = 200

# ICAの分離フィルタを初期化
Wica = np.zeros(shape=(Nk, n_sources, n_sources), dtype=np.complex)

Wica = Wica + np.eye(n_sources)[None, ...]

Wiva = Wica.copy()
Wiva_ip = Wica.copy()

start_time = time.time()
# 自然勾配法に基づくIVA実行コード（引数に与える関数を変更するだけ)
Wiva, s_iva, cost_buff_iva = execute_natural_gradient_ica(stft_data, Wiva, phi_func=phi_multivariate_laplacian,
                                                          contrast_func=contrast_multivariate_laplacian, mu=0.1,
                                                          n_ica_iterations=n_ica_iterations, is_use_non_holonomic=False)
y_iva = projection_back(s_iva, Wiva)
iva_time = time.time()

# IP法に基づくIVA実行コード（引数に与える関数を変更するだけ)
Wiva_ip, s_iva_ip, cost_buff_iva_ip = execute_ip_multivariate_laplacian_iva(stft_data, Wiva_ip,
                                                                            n_iterations=n_ica_iterations)
y_iva_ip = projection_back(s_iva_ip, Wiva_ip)
iva_ip_time = time.time()

Wica, s_ica, cost_buff_ica = execute_natural_gradient_ica(stft_data, Wica, mu=0.1, n_ica_iterations=n_ica_iterations,
                                                          is_use_non_holonomic=False)
permutation_index_result = solver_inter_frequency_permutation(s_ica)
y_ica = projection_back(s_ica, Wica)

# パーミュテーションを解く
for k in range(Nk):
    y_ica[:, :, k, :] = y_ica[:, permutation_index_result[k], k, :]

ica_time = time.time()

t, y_ica = sp.istft(y_ica[0, ...], fs=sample_rate, window="hann", nperseg=N)
t, y_iva = sp.istft(y_iva[0, ...], fs=sample_rate, window="hann", nperseg=N)
t, y_iva_ip = sp.istft(y_iva_ip[0, ...], fs=sample_rate, window="hann", nperseg=N)

write_file_from_time_signal(y_ica[0, ...] * np.iinfo(np.int16).max, "./real_{}_1.wav".format(wave_name), sample_rate)
make_Spectrogram("./real_{}_1.wav".format(wave_name), "./spectrogram_real_{}_1".format(wave_name))
make_Waveform("./real_{}_1.wav".format(wave_name), "./wave_real_{}_1".format(wave_name))
write_file_from_time_signal(y_ica[1, ...] * np.iinfo(np.int16).max, "./real_{}_2.wav".format(wave_name), sample_rate)
make_Spectrogram("./real_{}_2.wav".format(wave_name), "./spectrogram_real_{}_2".format(wave_name))
make_Waveform("./real_{}_2.wav".format(wave_name), "./wave_real_{}_2".format(wave_name))

write_file_from_time_signal(y_iva[0, ...] * np.iinfo(np.int16).max, "./real_iva_{}_1.wav".format(wave_name), sample_rate)
make_Spectrogram("./real_iva_{}_1.wav".format(wave_name), "./spectrogram_real_iva_{}_1".format(wave_name))
make_Waveform("./real_iva_{}_1.wav".format(wave_name), "./wave_real_iva_{}_1".format(wave_name))
write_file_from_time_signal(y_iva[1, ...] * np.iinfo(np.int16).max, "./real_iva_{}_2.wav".format(wave_name), sample_rate)
make_Spectrogram("./real_iva_{}_2.wav".format(wave_name), "./spectrogram_real_iva_{}_2".format(wave_name))
make_Waveform("./real_iva_{}_2.wav".format(wave_name), "./wave_real_iva_{}_2".format(wave_name))

write_file_from_time_signal(y_iva_ip[0, ...] * np.iinfo(np.int16).max, "./real_iva_ip_{}_1.wav".format(wave_name), sample_rate)
make_Spectrogram("./real_iva_ip_{}_1.wav".format(wave_name), "./spectrogram_real_iva_ip_{}_1".format(wave_name))
make_Waveform("./real_iva_ip_{}_1.wav".format(wave_name), "./wave_real_iva_ip_{}_1".format(wave_name))
write_file_from_time_signal(y_iva_ip[1, ...] * np.iinfo(np.int16).max, "./real_iva_ip_{}_2.wav".format(wave_name), sample_rate)
make_Spectrogram("./real_iva_ip_{}_2.wav".format(wave_name), "./spectrogram_real_iva_ip_{}_2".format(wave_name))
make_Waveform("./real_iva_ip_{}_2.wav".format(wave_name), "./wave_real_iva_ip_{}_2".format(wave_name))


#--------------再生-------------------

verification = input("分離結果を再生しますか(y/n)")

if verification != "n":
    PlayWavFie("./real_iva_ip_{}_1.wav".format(wave_name))
    PlayWavFie("./real_iva_ip_{}_2.wav".format(wave_name))

    print("fin")

