import io
import numpy as np
import streamlit as st
import soundfile as sf
import librosa

# Import explícito con alias para evitar colisiones de nombres
from librosa.effects import time_stretch as lb_time_stretch
from librosa.effects import pitch_shift as lb_pitch_shift

# -----------------------------
# Helpers
# -----------------------------
MAJOR_KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def load_audio(file, sr: int = 44100) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(file, sr=sr, mono=True)
    return y.astype(np.float32), sr

def estimate_bpm(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo) if tempo and tempo > 0 else 120.0

def estimate_key(y: np.ndarray, sr: int) -> tuple[str, int]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = chroma.mean(axis=1)

    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

    # Correlaciones circulares simples
    corr_major = np.array([np.dot(np.roll(profile, -k), major_profile) for k in range(12)])
    corr_minor = np.array([np.dot(np.roll(profile, -k), minor_profile) for k in range(12)])

    if corr_major.max() >= corr_minor.max():
        pc = int(np.argmax(corr_major))
        return f"{MAJOR_KEYS[pc]} Major", pc
    else:
        pc = int(np.argmax(corr_minor))
        return f"{MAJOR_KEYS[pc]} Minor", pc

def semitones_to_shift(src_pc: int, tgt_pc: int) -> int:
    d = tgt_pc - src_pc
    if d > 6: d -= 12
    if d < -6: d += 12
    return int(d)

# Fallback propio por si lb_time_stretch tiene conflicto en el entorno
def _pv_time_stretch(y: np.ndarray, rate: float, hop_length: int = 512, win_length: int = 2048) -> np.ndarray:
    if rate <= 0: return y
    stft = librosa.stft(y, n_fft=win_length, hop_length=hop_length, win_length=win_length)
    stretched = librosa.phase_vocoder(stft, rate=rate, hop_length=hop_length)
    y_out = librosa.istft(stretched, hop_length=hop_length, win_length=win_length, length=int(round(len(y)/rate)))
    return y_out.astype(np.float32)

def apply_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    if rate <= 0 or np.isclose(rate, 1.0): return y
    try:
        return lb_time_stretch(y, rate).astype(np.float32)
    except Exception:
        # Conflictos de nombres o errores internos → fallback
        return _pv_time_stretch(y, rate)

def apply_pitch_shift(y: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    if abs(n_semitones) <= 1e-6: return y
    try:
        return lb_pitch_shift(y, sr=sr, n_steps=n_semitones).astype(np.float32)
    except Exception:
        # Fallback simple: resample pitch (variará duración); luego reestira al original
        factor = 2.0 ** (n_semitones / 12.0)
        y_ps = librosa.resample(y, orig_sr=sr, target_sr=int(round(sr*factor)))
        return apply_time_stretch(y_ps, rate=factor)  # re-igualar duración aprox.

def align_to_target(y: np.ndarray, sr: int, bpm: float, key_pc: int,
                    target_bpm: float, target_key_pc: int | None,
                    do_time: bool, do_pitch: bool) -> np.ndarray:
    x = y
    if do_time and bpm > 0:
        rate = float(target_bpm) / float(bpm)
        x = apply_time_stretch(x, rate)
    if do_pitch and target_key_pc is not None:
        x = apply_pitch_shift(x, sr, semitones_to_shift(key_pc, target_key_pc))
    return x

def make_equal_power_fade(n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, num=n, dtype=np.float32)
    return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)

def _samps(sec: float, sr: int) -> int:
    return max(0, int(round(float(sec) * sr)))

def crossfade_tracks(a: np.ndarray, b: np.ndarray, sr: int, overlap_sec: float = 10.0,
                     a_mix_start: float = 0.0, b_mix_start: float = 0.0) -> np.ndarray:
    a_start = min(len(a), _samps(a_mix_start, sr))
    b_start = min(len(b), _samps(b_mix_start, sr))
    fade_len = _samps(overlap_sec, sr)

    # Longitudes disponibles para el solapamiento
    n = min(fade_len, len(a) - a_start, len(b) - b_start)
    if n <= 0:
        return np.concatenate([a, b])

    a_seg = a[a_start:a_start + n]
    b_seg = b[b_start:b_start + n]

    fade_out, fade_in = make_equal_power_fade(n)
    mixed_overlap = a_seg * fade_out + b_seg * fade_in

    out = np.concatenate([
        a[:a_start],
        mixed_overlap,
        b[b_start + n:],
    ])
    return out

def add_clean_segments(a: np.ndarray, b: np.ndarray, sr: int, extra_sec: int = 10,
                       a_mix_start: float = 0.0, b_mix_start: float = 0.0,
                       overlap_sec: float = 10.0) -> np.ndarray:
    pre_len = min(len(a), _samps(extra_sec, sr))
    post_len = min(len(b), _samps(extra_sec, sr))
    extra_a = a[len(a)-pre_len:]
    extra_b = b[:post_len]
    core = crossfade_tracks(a, b, sr, overlap_sec=overlap_sec,
                            a_mix_start=a_mix_start, b_mix_start=b_mix_start)
    y = np.concatenate([extra_a, core, extra_b])
    # Normalización suave para evitar clipping
    peak = np.max(np.abs(y)) if y.size else 1.0
    if peak > 0.99:
        y = (0.99 / peak) * y
    return y.astype(np.float32)

def write_wav_to_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    buf.seek(0)
    return buf.read()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="DJ Mixer AI", page_icon="🎧", layout="wide")
st.title("🎧 Mezclador DJ con IA (Archivos locales)")
st.write("Sube dos canciones y el sistema hará una mezcla con crossfade. "
         "Elige en qué segundo de cada track comienza la mezcla.")

with st.sidebar:
    st.header("Opciones")
    target_bpm = st.slider("BPM objetivo", 60, 180, 124)
    align_tempo = st.checkbox("Alinear tempo", value=True)
    align_key = st.checkbox("Alinear tono (pitch)", value=True)
    extra_sec = st.slider("Segundos extra de A y B", 5, 20, 10)
    overlap_sec = st.slider("Solapamiento (crossfade) [s]", 3, 20, 10)
    a_mix_start = st.number_input("Inicio de mezcla en track A (seg)", min_value=0.0, value=30.0, step=1.0)
    b_mix_start = st.number_input("Inicio de mezcla en track B (seg)", min_value=0.0, value=0.0, step=1.0)

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Canción A (WAV/MP3/OGG/FLAC)", type=["wav","mp3","ogg","flac"], key="a")
with col2:
    file_b = st.file_uploader("Canción B (WAV/MP3/OGG/FLAC)", type=["wav","mp3","ogg","flac"], key="b")

# Análisis inmediato de cada track
track_info = {}
for label, file in zip(["A","B"], [file_a, file_b]):
    if file:
        y, sr = load_audio(file)
        bpm = estimate_bpm(y, sr)
        key_label, key_pc = estimate_key(y, sr)
        track_info[label] = dict(y=y, sr=sr, bpm=bpm, key=key_label, pc=key_pc)
        st.sidebar.markdown(f"### Track {label}")
        st.sidebar.metric(f"BPM {label}", f"{bpm:.1f}")
        st.sidebar.caption(f"Clave {label}: {key_label}")

if file_a and file_b and st.button("Mezclar 🎶"):
    with st.spinner("Procesando tracks..."):
        y_a = track_info["A"]["y"]; y_b = track_info["B"]["y"]
        sr = track_info["A"]["sr"]
        # Si suben A y B con SR distinto, re-muestrea B a SR de A
        if track_info["B"]["sr"] != sr:
            y_b = librosa.resample(y_b, orig_sr=track_info["B"]["sr"], target_sr=sr)

        bpm_a = track_info["A"]["bpm"]; bpm_b = track_info["B"]["bpm"]
        key_pc_a = track_info["A"]["pc"]; key_pc_b = track_info["B"]["pc"]
        key_label_a = track_info["A"]["key"]; key_label_b = track_info["B"]["key"]

        target_key_pc = key_pc_a if align_key else None

        y_a_aligned = align_to_target(y_a, sr, bpm_a, key_pc_a, target_bpm, target_key_pc, align_tempo, align_key)
        y_b_aligned = align_to_target(y_b, sr, bpm_b, key_pc_b, target_bpm, target_key_pc, align_tempo, align_key)

        mixed = add_clean_segments(y_a_aligned, y_b_aligned, sr, extra_sec=extra_sec,
                                   a_mix_start=a_mix_start, b_mix_start=b_mix_start,
                                   overlap_sec=overlap_sec)

    st.markdown("### Mezcla final")
    wav_bytes = write_wav_to_bytes(mixed, sr)
    st.audio(wav_bytes, format="audio/wav")
    st.download_button("⬇️ Descargar mezcla (WAV)", data=wav_bytes,
                       file_name="mezcla_dj_ai.wav", mime="audio/wav")
else:
    st.info("👆 Sube dos archivos de audio para comenzar.")
