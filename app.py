import io
import numpy as np
import streamlit as st
import soundfile as sf
import librosa

# -----------------------------
# Helpers
# -----------------------------
MAJOR_KEYS = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]

def db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20 * np.log10(np.maximum(eps, np.abs(x)))


def load_audio(file, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Carga archivo de audio en mono, 44.1 kHz"""
    y, sr = librosa.load(file, sr=sr, mono=True)
    return y.astype(np.float32), sr


def estimate_bpm(y: np.ndarray, sr: int) -> float:
    """Estimación básica de BPM"""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo) if tempo > 0 else 120.0


def estimate_key(y: np.ndarray, sr: int) -> tuple[str, int]:
    """Estimación de tonalidad usando perfiles de Krumhansl"""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = chroma.mean(axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    corr_major = np.correlate(profile, major_profile, mode="same")
    corr_minor = np.correlate(profile, minor_profile, mode="same")

    if corr_major.max() > corr_minor.max():
        pitch_class = int(np.argmax(corr_major) % 12)
        return MAJOR_KEYS[pitch_class] + " Major", pitch_class
    else:
        pitch_class = int(np.argmax(corr_minor) % 12)
        return MAJOR_KEYS[pitch_class] + " Minor", pitch_class


def semitones_to_shift(src_pc: int, tgt_pc: int) -> int:
    """Diferencia en semitonos (ajustada a ±6)"""
    d = tgt_pc - src_pc
    if d > 6:
        d -= 12
    if d < -6:
        d += 12
    return int(d)


def apply_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Estiramiento temporal"""
    return librosa.effects.time_stretch(y, rate) if rate > 0 else y


def apply_pitch_shift(y: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    """Cambio de tono"""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_semitones) if abs(n_semitones) > 1e-6 else y


def align_to_target(y: np.ndarray, sr: int, bpm: float, key_pc: int,
                    target_bpm: float, target_key_pc: int | None,
                    do_time: bool, do_pitch: bool) -> np.ndarray:
    """Alinea track a tempo y tonalidad destino"""
    x = y
    if do_time and bpm > 0:
        x = apply_time_stretch(x, target_bpm / bpm)
    if do_pitch and target_key_pc is not None:
        x = apply_pitch_shift(x, sr, semitones_to_shift(key_pc, target_key_pc))
    return x


def make_equal_power_fade(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Curvas cos/sin para crossfade de potencia igual"""
    t = np.linspace(0, 1, num=n, dtype=np.float32)
    return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)


def crossfade_tracks(a: np.ndarray, b: np.ndarray, sr: int, overlap_sec: float = 10.0,
                     a_mix_start: float = 0.0, b_mix_start: float = 0.0) -> np.ndarray:
    """Crossfade natural entre dos tracks"""
    a_start = int(a_mix_start * sr)
    b_start = int(b_mix_start * sr)
    fade_len = int(overlap_sec * sr)

    a_seg = a[a_start:a_start + fade_len]
    b_seg = b[b_start:b_start + fade_len]
    n = min(len(a_seg), len(b_seg))
    if n <= 0:
        return np.concatenate([a, b])

    fade_out, fade_in = make_equal_power_fade(n)
    mixed_overlap = a_seg[:n] * fade_out + b_seg[:n] * fade_in

    return np.concatenate([
        a[:a_start],         # parte inicial de A
        mixed_overlap,        # solapamiento
        b[b_start + n:],      # resto de B
    ])


def add_clean_segments(a: np.ndarray, b: np.ndarray, sr: int, extra_sec: int = 10,
                       a_mix_start: float = 0.0, b_mix_start: float = 0.0) -> np.ndarray:
    """Añade márgenes extra de A y B para evitar cortes bruscos"""
    extra_a = a[-extra_sec * sr:]
    extra_b = b[:extra_sec * sr]
    return np.concatenate([extra_a,
                           crossfade_tracks(a, b, sr, overlap_sec=10.0,
                                            a_mix_start=a_mix_start,
                                            b_mix_start=b_mix_start),
                           extra_b])


def write_wav_to_bytes(y: np.ndarray, sr: int) -> bytes:
    """Exporta audio como WAV en memoria"""
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
         "Puedes elegir en qué segundo de cada track comienza la mezcla.")

with st.sidebar:
    st.header("Opciones")
    target_bpm = st.slider("BPM objetivo", 60, 180, 124)
    align_tempo = st.checkbox("Alinear tempo", value=True)
    align_key = st.checkbox("Alinear tono (pitch)", value=True)
    extra_sec = st.slider("Segundos extra de A y B", 5, 20, 10)
    a_mix_start = st.number_input("Inicio de mezcla en track A (seg)", min_value=0, value=30)
    b_mix_start = st.number_input("Inicio de mezcla en track B (seg)", min_value=0, value=0)

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Canción A (WAV/MP3/OGG)", type=["wav", "mp3", "ogg", "flac"], key="a")
with col2:
    file_b = st.file_uploader("Canción B (WAV/MP3/OGG)", type=["wav", "mp3", "ogg", "flac"], key="b")

# Mostrar análisis inmediato de cada track
track_info = {}
for label, file in zip(["A", "B"], [file_a, file_b]):
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
        y_a = track_info["A"]["y"]
        y_b = track_info["B"]["y"]
        sr = track_info["A"]["sr"]

        bpm_a = track_info["A"]["bpm"]
        bpm_b = track_info["B"]["bpm"]
        key_pc_a = track_info["A"]["pc"]
        key_pc_b = track_info["B"]["pc"]
        key_label_a = track_info["A"]["key"]
        key_label_b = track_info["B"]["key"]

        target_key_pc = key_pc_a if align_key else None

        y_a_aligned = align_to_target(y_a, sr, bpm_a, key_pc_a, target_bpm, target_key_pc, align_tempo, align_key)
        y_b_aligned = align_to_target(y_b, sr, bpm_b, key_pc_b, target_bpm, target_key_pc, align_tempo, align_key)

        mixed = add_clean_segments(y_a_aligned, y_b_aligned, sr, extra_sec=extra_sec,
                                   a_mix_start=a_mix_start, b_mix_start=b_mix_start)

    st.markdown("### Mezcla final")
    wav_bytes = write_wav_to_bytes(mixed, sr)
    st.audio(wav_bytes, format="audio/wav")
    st.download_button("⬇️ Descargar mezcla (WAV)", data=wav_bytes,
                       file_name="mezcla_dj_ai.wav", mime="audio/wav")
else:
    st.info("👆 Sube dos archivos de audio para comenzar.")
