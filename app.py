import io
import os
import tempfile
from typing import Tuple, Optional

import numpy as np
import streamlit as st
import soundfile as sf
import librosa
import yt_dlp

# -----------------------------
# Helpers
# -----------------------------
MAJOR_KEYS = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]

def db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20 * np.log10(np.maximum(eps, np.abs(x)))


def download_audio_from_youtube(url: str, sr: int = 44100) -> Tuple[np.ndarray, int]:
    """Descargar audio desde YouTube y devolver como waveform."""
    tmpdir = tempfile.mkdtemp()
    outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".wav"
    y, sr = librosa.load(filename, sr=sr, mono=True)
    return y.astype(np.float32), sr


def estimate_bpm(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo) if tempo > 0 else 120.0


def estimate_key(y: np.ndarray, sr: int) -> Tuple[str, int]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = chroma.mean(axis=1)
    pitch_class = int(np.argmax(profile))
    return MAJOR_KEYS[pitch_class] + " (approx)", pitch_class


def semitones_to_shift(src_pc: int, tgt_pc: int) -> int:
    d = tgt_pc - src_pc
    if d > 6:
        d -= 12
    if d < -6:
        d += 12
    return int(d)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate) if rate > 0 else y


def pitch_shift(y: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_semitones) if abs(n_semitones) > 1e-6 else y


def align_to_target(y: np.ndarray, sr: int, bpm: float, key_pc: int,
                    target_bpm: float, target_key_pc: Optional[int],
                    do_time: bool, do_pitch: bool) -> np.ndarray:
    x = y
    if do_time and bpm > 0:
        x = time_stretch(x, target_bpm / bpm)
    if do_pitch and target_key_pc is not None:
        x = pitch_shift(x, sr, semitones_to_shift(key_pc, target_key_pc))
    return x


def make_equal_power_fade(n: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, num=n, dtype=np.float32)
    return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)


def crossfade_tracks(a: np.ndarray, b: np.ndarray, sr: int, overlap_sec: float = 10.0) -> np.ndarray:
    fade_len = int(overlap_sec * sr)
    n = min(len(a), len(b), fade_len)
    fade_out, fade_in = make_equal_power_fade(n)
    a_seg = a[-n:] * fade_out
    b_seg = b[:n] * fade_in
    mixed_overlap = a_seg + b_seg
    return np.concatenate([a[:-n], mixed_overlap, b[n:]])


def add_clean_segments(a: np.ndarray, b: np.ndarray, sr: int, extra_sec: int = 10) -> np.ndarray:
    extra_a = a[-extra_sec * sr:]
    extra_b = b[:extra_sec * sr]
    return np.concatenate([extra_a, crossfade_tracks(a, b, sr), extra_b])


def write_wav_to_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    buf.seek(0)
    return buf.read()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="DJ Mixer AI", page_icon="🎧", layout="wide")

st.title("🎧 Mezclador DJ con IA (YouTube Edition)")
st.write("Ingresa dos links de YouTube y el sistema descargará, analizará y mezclará los tracks.")

with st.sidebar:
    st.header("Opciones")
    target_bpm = st.slider("BPM objetivo", 60, 180, 124)
    align_tempo = st.checkbox("Alinear tempo", value=True)
    align_key = st.checkbox("Alinear tono (pitch)", value=True)
    extra_sec = st.slider("Segundos extra de A y B", 5, 20, 10)

url_a = st.text_input("Link YouTube - Canción A")
url_b = st.text_input("Link YouTube - Canción B")

if url_a and url_b:
    if st.button("Mezclar 🎶"):
        with st.spinner("Descargando y mezclando..."):
            y_a, sr = download_audio_from_youtube(url_a)
            y_b, _ = download_audio_from_youtube(url_b)

            bpm_a = estimate_bpm(y_a, sr)
            bpm_b = estimate_bpm(y_b, sr)
            key_label_a, key_pc_a = estimate_key(y_a, sr)
            key_label_b, key_pc_b = estimate_key(y_b, sr)

            target_key_pc = key_pc_a if align_key else None

            y_a_aligned = align_to_target(y_a, sr, bpm_a, key_pc_a, target_bpm, target_key_pc, align_tempo, align_key)
            y_b_aligned = align_to_target(y_b, sr, bpm_b, key_pc_b, target_bpm, target_key_pc, align_tempo, align_key)

            mixed = add_clean_segments(y_a_aligned, y_b_aligned, sr, extra_sec=extra_sec)

        st.markdown("### Análisis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BPM A", f"{bpm_a:.1f}")
            st.caption(f"Clave A: {key_label_a}")
        with col2:
            st.metric("BPM B", f"{bpm_b:.1f}")
            st.caption(f"Clave B: {key_label_b}")
        with col3:
            st.metric("BPM objetivo", f"{target_bpm}")

        st.markdown("### Mezcla final")
        wav_bytes = write_wav_to_bytes(mixed, sr)
        st.audio(wav_bytes, format="audio/wav")
        st.download_button("⬇️ Descargar mezcla (WAV)", data=wav_bytes,
                           file_name="mezcla_dj_ai.wav", mime="audio/wav")

else:
    st.info("👆 Ingresa dos links de YouTube para comenzar.")
