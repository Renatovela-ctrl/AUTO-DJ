import io
import numpy as np
import streamlit as st
import soundfile as sf
import librosa
from librosa.effects import time_stretch as lb_time_stretch, pitch_shift as lb_pitch_shift

# Configuración inicial
MAJOR_KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
DEFAULT_SR = 44100

def load_audio(file, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Carga el audio con manejo de errores mejorado"""
    try:
        y, sr = librosa.load(file, sr=sr, mono=True)
        return y.astype(np.float32), sr
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        raise

def estimate_bpm(y: np.ndarray, sr: int) -> float:
    """Estimación de BPM más robusta"""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)
        return float(tempo) if tempo and tempo > 0 else 120.0
    except:
        return 120.0

def estimate_key(y: np.ndarray, sr: int) -> tuple[str, int]:
    """Estimación de clave musical mejorada"""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        profile = chroma.mean(axis=1)
        
        major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        
        corr_major = np.array([np.dot(np.roll(profile, -k), major_profile) for k in range(12)])
        corr_minor = np.array([np.dot(np.roll(profile, -k), minor_profile) for k in range(12)])
        
        if corr_major.max() >= corr_minor.max():
            pc = int(np.argmax(corr_major))
            return f"{MAJOR_KEYS[pc]} Major", pc
        else:
            pc = int(np.argmax(corr_minor))
            return f"{MAJOR_KEYS[pc]} Minor", pc
    except:
        return "C Major", 0

def beat_sync_mix(y1: np.ndarray, y2: np.ndarray, sr: int, bpm: float, overlap_beats: int = 8) -> np.ndarray:
    """Mezcla sincronizada por beats para transiciones más profesionales"""
    beat_length = 60 / bpm  # duración de un beat en segundos
    beat_samples = int(beat_length * sr)
    
    # Encontrar el primer beat fuerte para alinear
    onset_env = librosa.onset.onset_strength(y=y1, sr=sr)
    beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=bpm)[1]
    first_strong_beat = librosa.frames_to_samples(beats[0]) if len(beats) > 0 else 0
    
    # Asegurar que tenemos suficiente audio para la mezcla
    mix_start = first_strong_beat
    mix_duration = overlap_beats * beat_samples
    mix_end = mix_start + mix_duration
    
    if mix_end > len(y1) or mix_duration > len(y2):
        # Fallback a crossfade simple si no hay suficiente audio
        return simple_crossfade(y1, y2, sr, overlap_sec=8*beat_length)
    
    # Crear ventana de mezcla
    fade_out = np.sqrt(np.linspace(1, 0, mix_duration))  # curva sqrt para mejor transición
    fade_in = np.sqrt(np.linspace(0, 1, mix_duration))
    
    # Aplicar la mezcla
    mixed = np.concatenate([
        y1[:mix_start],
        y1[mix_start:mix_end] * fade_out + y2[:mix_duration] * fade_in,
        y2[mix_duration:]
    ])
    
    return mixed

def simple_crossfade(a: np.ndarray, b: np.ndarray, sr: int, overlap_sec: float = 8.0) -> np.ndarray:
    """Crossfade simple con curva mejorada"""
    overlap_samples = int(overlap_sec * sr)
    min_len = min(len(a), len(b), overlap_samples)
    
    if min_len <= 0:
        return np.concatenate([a, b])
    
    # Curva de fade mejorada (sqrt para transición más suave)
    fade_out = np.sqrt(np.linspace(1, 0, min_len))
    fade_in = np.sqrt(np.linspace(0, 1, min_len))
    
    return np.concatenate([
        a[:-min_len],
        a[-min_len:] * fade_out + b[:min_len] * fade_in,
        b[min_len:]
    ])

def write_wav_to_bytes(y: np.ndarray, sr: int) -> bytes:
    """Escribe el audio a bytes con normalización"""
    # Normalización de volumen
    peak = np.max(np.abs(y))
    if peak > 0.9:
        y = y * (0.9 / peak)
    
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    buf.seek(0)
    return buf.read()

# Interfaz de usuario mejorada
st.set_page_config(page_title="DJ Mixer Pro", page_icon="🎛️", layout="wide")
st.title("🎛️ Mezclador DJ Profesional")

with st.sidebar:
    st.header("⚙️ Configuración de mezcla")
    target_bpm = st.slider("BPM objetivo", 60, 180, 124)
    overlap_beats = st.slider("Beats de transición", 4, 16, 8)
    use_beat_sync = st.checkbox("Sincronización por beats", value=True)
    normalize = st.checkbox("Normalizar volumen", value=True)

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Track A", type=["wav","mp3","ogg","flac"], key="a")
with col2:
    file_b = st.file_uploader("Track B", type=["wav","mp3","ogg","flac"], key="b")

if file_a and file_b:
    with st.spinner("Procesando pistas..."):
        try:
            y_a, sr_a = load_audio(file_a)
            y_b, sr_b = load_audio(file_b)
            
            # Asegurar misma tasa de muestreo
            if sr_a != sr_b:
                y_b = librosa.resample(y_b, orig_sr=sr_b, target_sr=sr_a)
                sr_b = sr_a
            
            # Estimar BPM y clave
            bpm_a = estimate_bpm(y_a, sr_a)
            bpm_b = estimate_bpm(y_b, sr_b)
            
            # Mezcla profesional
            if use_beat_sync:
                mixed = beat_sync_mix(y_a, y_b, sr_a, target_bpm, overlap_beats)
            else:
                mixed = simple_crossfade(y_a, y_b, sr_a, overlap_sec=overlap_beats*(60/target_bpm))
            
            st.success("✅ Mezcla completada")
            st.audio(write_wav_to_bytes(mixed, sr_a), format='audio/wav')
            
        except Exception as e:
            st.error(f"Error durante la mezcla: {str(e)}")
