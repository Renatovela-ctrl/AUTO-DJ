import io
import os
import json
import requests
from typing import Tuple, Optional, Dict, Any
import numpy as np
import streamlit as st
import soundfile as sf
import librosa
from librosa.effects import time_stretch as lb_time_stretch
from librosa.effects import pitch_shift as lb_pitch_shift

# =========================
# Configuración general
# =========================
st.set_page_config(page_title="🎛️ DJ Mixer Pro AI", page_icon="🤖", layout="wide")

MAJOR_KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
DEFAULT_SR = 44100

# Modelos de Hugging Face
HF_MODEL_GENRE = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
HF_MODEL_STRUCTURE = "mtg/essentia-musicstructure"
HF_MODEL_ONSET = "audioset/onset-detection"

# =========================
# Utilidades de autenticación
# =========================
def get_hf_token() -> Optional[str]:
    # 1) Secrets de Streamlit
    token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None) if hasattr(st, "secrets") else None
    if token: 
        return token
    # 2) Variable de entorno
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)
    if token:
        return token
    # 3) Token de respaldo (¡No usar en producción!)
    return "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"

# =========================
# Audio core
# =========================
def load_audio(file, sr: int = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    try:
        y, sr = librosa.load(file, sr=sr, mono=True)
        return y.astype(np.float32), sr
    except Exception as e:
        st.error(f"Error al cargar audio: {e}")
        return np.array([], dtype=np.float32), sr

def estimate_bpm(y: np.ndarray, sr: int) -> float:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item() if tempo.size > 0 else 120.0
        return float(tempo) if tempo and tempo > 0 else 120.0
    except Exception as e:
        st.warning(f"No se pudo estimar BPM, uso 120. Detalle: {e}")
        return 120.0

def estimate_key(y: np.ndarray, sr: int) -> Tuple[str, int]:
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        if chroma.size == 0:
            return "C Major", 0
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
    except Exception as e:
        st.warning(f"No se pudo estimar la tonalidad, usando C Major. Detalle: {e}")
        return "C Major", 0

def energy_score(y: np.ndarray, sr: int) -> float:
    """Puntaje simple de energía (RMS medio * centroid medio)."""
    try:
        rms = librosa.feature.rms(y=y).mean()
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        return float(rms * centroid)
    except Exception:
        return 0.0

def semitones_to_shift(src_pc: int, tgt_pc: int) -> int:
    d = tgt_pc - src_pc
    if d > 6: d -= 12
    if d < -6: d += 12
    return int(d)

def apply_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    if rate <= 0 or np.isclose(rate, 1.0):
        return y
    try:
        return lb_time_stretch(y, rate=rate).astype(np.float32)
    except Exception as e:
        st.warning(f"Fallo time_stretch, phase vocoder. {e}")
        hop_length = 512
        stft = librosa.stft(y, hop_length=hop_length)
        stft_stretch = librosa.phase_vocoder(stft, rate=rate, hop_length=hop_length)
        out = librosa.istft(stft_stretch, hop_length=hop_length)
        return out.astype(np.float32)

def apply_pitch_shift(y: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    if abs(n_semitones) <= 1e-6:
        return y
    try:
        return lb_pitch_shift(y, sr=sr, n_steps=n_semitones).astype(np.float32)
    except Exception as e:
        st.warning(f"Fallo pitch_shift, usando resample. {e}")
        factor = 2.0 ** (n_semitones / 12.0)
        target_sr = int(sr * factor)
        y_rs = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return apply_time_stretch(y_rs, rate=1.0 / factor)

def align_to_target(y: np.ndarray, sr: int, bpm: float, key_pc: int,
                    target_bpm: float, target_key_pc: Optional[int],
                    do_time: bool, do_pitch: bool) -> np.ndarray:
    x = y
    if do_time and bpm > 0:
        rate = float(target_bpm) / float(bpm)
        x = apply_time_stretch(x, rate)
    if do_pitch and target_key_pc is not None:
        shift = semitones_to_shift(key_pc, target_key_pc)
        x = apply_pitch_shift(x, sr, shift)
    return x

def find_beat_position(y: np.ndarray, sr: int, start_time: float) -> float:
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=120)
        if beats is None or len(beats) == 0:
            return start_time
        beat_times = librosa.frames_to_time(beats, sr=sr)
        return float(min(beat_times, key=lambda x: abs(x - start_time)))
    except Exception as e:
        st.warning(f"No se pudo localizar beat exacto. {e}")
        return start_time

def beat_sync_crossfade(a: np.ndarray, b: np.ndarray, sr: int,
                        a_start: float, b_start: float,
                        overlap_beats: int = 8) -> np.ndarray:
    a_beat_start = find_beat_position(a, sr, a_start)
    b_beat_start = find_beat_position(b, sr, b_start)

    a_start_samp = int(a_beat_start * sr)
    b_start_samp = int(b_beat_start * sr)

    tempo = estimate_bpm(a, sr)
    beat_duration = 60.0 / max(tempo, 1e-6)
    overlap_samples = int(max(1, overlap_beats) * beat_duration * sr)

    n = min(overlap_samples, len(a) - a_start_samp, len(b) - b_start_samp)
    if n <= 0:
        return np.concatenate([a[:a_start_samp], b[b_start_samp:]])

    t = np.linspace(0, 1, n, endpoint=False)
    fade_out = np.sqrt(1.0 - t)
    fade_in  = np.sqrt(t)

    a_seg = a[a_start_samp:a_start_samp + n]
    b_seg = b[b_start_samp:b_start_samp + n]
    mixed_overlap = a_seg * fade_out + b_seg * fade_in

    return np.concatenate([a[:a_start_samp], mixed_overlap, b[b_start_samp + n:]])

def time_crossfade(a: np.ndarray, b: np.ndarray, sr: int,
                   a_start: float, b_start: float,
                   overlap_seconds: float = 8.0) -> np.ndarray:
    a_start_samp = int(a_start * sr)
    b_start_samp = int(b_start * sr)
    overlap_samples = int(max(0.1, overlap_seconds) * sr)

    n = min(overlap_samples, len(a) - a_start_samp, len(b) - b_start_samp)
    if n <= 0:
        return np.concatenate([a[:a_start_samp], b[b_start_samp:]])

    t = np.linspace(0, 1, n, endpoint=False)
    fade_out = np.sqrt(1.0 - t)
    fade_in  = np.sqrt(t)

    a_seg = a[a_start_samp:a_start_samp + n]
    b_seg = b[b_start_samp:b_start_samp + n]
    mixed_overlap = a_seg * fade_out + b_seg * fade_in

    return np.concatenate([a[:a_start_samp], mixed_overlap, b[b_start_samp + n:]])

def write_wav_to_bytes(y: np.ndarray, sr: int) -> bytes:
    if y.size == 0:
        return b""
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = (y * (0.99 / peak)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf.getvalue()

def export_wav_bytes_from_array(y: np.ndarray, sr: int) -> bytes:
    """Convierte array a WAV bytes (para enviar a HF)."""
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()

# =========================
# Hugging Face Inference API
# =========================
def hf_audio_classify(file_bytes: bytes, model: str, token: str, timeout: int = 60) -> Any:
    """Llama al endpoint de inference API para audio-classification."""
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(url, headers=headers, data=file_bytes, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        else:
            try:
                return {"error": True, "status": resp.status_code, "detail": resp.json()}
            except Exception:
                return {"error": True, "status": resp.status_code, "detail": resp.text}
    except Exception as e:
        return {"error": True, "status": -1, "detail": str(e)}

def hf_audio_structure(file_bytes: bytes, token: str, timeout: int = 60) -> Any:
    """Obtiene estructura musical de la pista."""
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL_STRUCTURE}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(url, headers=headers, data=file_bytes, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": True, "status": resp.status_code, "detail": resp.text}
    except Exception as e:
        return {"error": True, "detail": str(e)}

def hf_audio_onsets(file_bytes: bytes, token: str, timeout: int = 60) -> Any:
    """Detecta puntos de transición óptimos."""
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL_ONSET}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(url, headers=headers, data=file_bytes, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": True, "status": resp.status_code, "detail": resp.text}
    except Exception as e:
        return {"error": True, "detail": str(e)}

# =========================
# Lógica DJ: recomendación IA
# =========================
def key_compatibility(pc_a: int, pc_b: int) -> int:
    """Compatibilidad de clave (0 = igual; 1 = +/-1; 2 = +/-2; 3 = otros)."""
    diff = min((pc_a - pc_b) % 12, (pc_b - pc_a) % 12)
    return int(diff)

def recommend_order(infoA: Dict, infoB: Dict) -> str:
    bpm_a, bpm_b = infoA["bpm"], infoB["bpm"]
    pc_a, pc_b = infoA["pc"], infoB["pc"]
    e_a, e_b   = infoA["energy"], infoB["energy"]

    key_diff = key_compatibility(pc_a, pc_b)
    bpm_diff_ab = abs(bpm_a - bpm_b)
    energy_up = (e_b >= e_a)
    penalty_key = 2 if key_diff >= 3 else (1 if key_diff == 2 else 0)
    score_ab = bpm_diff_ab + penalty_key - (0.5 if energy_up else 0)
    score_ba = bpm_diff_ab + penalty_key - (0.5 if (e_a >= e_b) else 0)

    return "A→B" if score_ab <= score_ba else "B→A"

def recommend_mix_params(infoA: Dict, infoB: Dict) -> Dict[str, Any]:
    """Recomienda parámetros de mezcla basados en IA."""
    # BPM objetivo: promedio ponderado por energía
    total_energy = infoA["energy"] + infoB["energy"]
    target_bpm = (infoA["bpm"] * infoA["energy"] + infoB["bpm"] * infoB["energy"]) / total_energy
    
    # Duración de transición basada en género
    genreA = infoA.get("top_genre", "").lower()
    genreB = infoB.get("top_genre", "").lower()
    
    electronic_genres = {"techno", "house", "trance", "edm", "dubstep"}
    if genreA in electronic_genres or genreB in electronic_genres:
        overlap = 4.0  # Transiciones rápidas para electrónica
    else:
        overlap = 8.0  # Transiciones más largas para otros géneros
    
    # Método de mezcla recomendado
    mix_method = "Sincronización por Beats" if abs(infoA["bpm"] - infoB["bpm"]) < 10 else "Crossfade Simple (seg)"
    
    return {
        "target_bpm": max(60, min(180, round(target_bpm))),
        "mix_method": mix_method,
        "overlap": overlap,
        "align_key": True,
        "align_tempo": True
    }

def find_optimal_mix_points(structA: Dict, structB: Dict) -> Tuple[float, float]:
    """Encuentra puntos óptimos de mezcla basados en estructura musical."""
    def find_section(sections, min_time=30.0, preferred_labels=["drop", "chorus"]):
        for section in sections:
            start = section.get("start", 0)
            label = section.get("label", "").lower()
            if start >= min_time:
                if any(p in label for p in preferred_labels):
                    return start
        # Si no encuentra sección preferida, usa la primera después de min_time
        for section in sections:
            if section.get("start", 0) >= min_time:
                return section.get("start", min_time)
        return min_time
    
    sectionsA = structA.get("sections", [])
    sectionsB = structB.get("sections", [])
    
    pointA = find_section(sectionsA, min_time=30.0)
    pointB = find_section(sectionsB, min_time=0.0)
    
    return pointA, pointB

# =========================
# UI
# =========================
st.title("🤖 DJ Mixer Pro AI")

# Inicializar estado de sesión
if "target_bpm" not in st.session_state:
    st.session_state.target_bpm = 124
if "a_mix_start" not in st.session_state:
    st.session_state.a_mix_start = 30.0
if "b_mix_start" not in st.session_state:
    st.session_state.b_mix_start = 0.0
if "mix_method" not in st.session_state:
    st.session_state.mix_method = "Sincronización por Beats"
if "overlap" not in st.session_state:
    st.session_state.overlap = 8
if "align_key" not in st.session_state:
    st.session_state.align_key = True
if "align_tempo" not in st.session_state:
    st.session_state.align_tempo = True

# Token HF
token = get_hf_token()

with st.sidebar:
    st.header("⚙️ Configuración")
    target_bpm = st.slider("BPM objetivo", 60, 180, st.session_state.target_bpm)
    align_tempo = st.checkbox("Alinear tempo", value=st.session_state.align_tempo)
    align_key = st.checkbox("Alinear tono (pitch)", value=st.session_state.align_key)

    st.subheader("Transición")
    mix_method = st.radio("Método de mezcla", 
                         ["Crossfade Simple (seg)", "Sincronización por Beats"], 
                         index=0 if st.session_state.mix_method == "Crossfade Simple (seg)" else 1)
    
    if mix_method == "Crossfade Simple (seg)":
        overlap = st.slider("Duración transición (seg)", 2.0, 20.0, float(st.session_state.overlap))
    else:
        overlap = st.slider("Beats de transición", 4, 32, st.session_state.overlap)

    st.subheader("Puntos de mezcla")
    a_mix_start = st.number_input("Inicio mezcla Track A (seg)", 
                                 min_value=0.0, 
                                 value=st.session_state.a_mix_start, 
                                 step=1.0)
    b_mix_start = st.number_input("Inicio mezcla Track B (seg)", 
                                 min_value=0.0, 
                                 value=st.session_state.b_mix_start, 
                                 step=1.0)

    extra_sec = st.slider("Audio extra al inicio/fin (seg)", 0, 30, 15)

    st.subheader("Optimización IA")
    if st.button("🧠 Optimizar con IA", type="primary", use_container_width=True):
        st.session_state.optimize_requested = True
    else:
        st.session_state.optimize_requested = False

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Track A", type=["wav","mp3","ogg","flac"], key="a")
with col2:
    file_b = st.file_uploader("Track B", type=["wav","mp3","ogg","flac"], key="b")

track_info = {}
structures = {"A": None, "B": None}

# Procesar A y B
for label, file in zip(["A", "B"], [file_a, file_b]):
    if file:
        # Cargar y mostrar
        y, sr = load_audio(file)
        if y.size == 0:
            continue
        with st.expander(f"🔎 Análisis Track {label}", expanded=True):
            st.audio(file, format="audio/wav")
            st.caption(f"Duración: {round(len(y)/sr, 2)} s | SR: {sr} Hz")

        # BPM, Key, Energía
        bpm = estimate_bpm(y, sr)
        key_label, key_pc = estimate_key(y, sr)
        eng = energy_score(y, sr)

        # Clasificación de género (HF)
        genre_result = None
        top_genre = ""
        if token:
            with st.spinner(f"Consultando Hugging Face para género del Track {label}..."):
                try:
                    wav_bytes = export_wav_bytes_from_array(y, sr)
                    genre_result = hf_audio_classify(wav_bytes, HF_MODEL_GENRE, token)
                    if isinstance(genre_result, list):
                        top_genre = genre_result[0].get("label", "")
                except Exception as e:
                    genre_result = {"error": True, "detail": str(e)}
        else:
            st.info("Para etiquetar género con Hugging Face, proporciona tu token.")

        # Guardar
        track_info[label] = dict(
            y=y, sr=sr, bpm=bpm, key=key_label, 
            pc=key_pc, energy=eng, genre=genre_result,
            top_genre=top_genre
        )

        # Mostrar
        st.sidebar.markdown(f"### Track {label}")
        st.sidebar.metric(f"BPM {label}", f"{bpm:.1f}")
        st.sidebar.caption(f"Tonalidad {label}: {key_label}")
        st.sidebar.caption(f"Energía {label}: {eng:.2f}")

        if isinstance(genre_result, list):
            top = sorted(genre_result, key=lambda x: x.get("score", 0), reverse=True)[:3]
            st.sidebar.caption(f"Género {label}: " + ", ".join([f"{it['label']} ({it['score']:.2f})" for it in top]))
        elif isinstance(genre_result, dict) and genre_result.get("error"):
            st.sidebar.caption(f"Género {label}: (error HF)")

        # Análisis de estructura si se solicitó optimización
        if token and st.session_state.optimize_requested:
            with st.spinner(f"Analizando estructura musical del Track {label}..."):
                try:
                    wav_bytes = export_wav_bytes_from_array(y, sr)
                    structures[label] = hf_audio_structure(wav_bytes, token)
                except Exception as e:
                    st.error(f"Error en análisis de estructura: {str(e)}")
                    structures[label] = {"error": True, "detail": str(e)}

# Optimización con IA
if st.session_state.optimize_requested and "A" in track_info and "B" in track_info:
    if structures["A"] and not structures["A"].get("error") and structures["B"] and not structures["B"].get("error"):
        try:
            # Encontrar puntos óptimos de mezcla
            pointA, pointB = find_optimal_mix_points(structures["A"], structures["B"])
            st.session_state.a_mix_start = pointA
            st.session_state.b_mix_start = pointB
            st.success(f"IA recomienda: Iniciar A en {pointA:.1f}s, B en {pointB:.1f}s")
        except Exception as e:
            st.warning(f"No se pudieron determinar puntos óptimos: {str(e)}")
    
    # Recomendar parámetros de mezcla
    rec_params = recommend_mix_params(track_info["A"], track_info["B"])
    st.session_state.target_bpm = rec_params["target_bpm"]
    st.session_state.mix_method = rec_params["mix_method"]
    st.session_state.overlap = rec_params["overlap"]
    st.session_state.align_key = rec_params["align_key"]
    st.session_state.align_tempo = rec_params["align_tempo"]
    
    st.success(f"IA recomienda: BPM objetivo {rec_params['target_bpm']}, "
               f"Método: {rec_params['mix_method']}, "
               f"Duración transición: {rec_params['overlap']}")

# Recomendación de orden
if "A" in track_info and "B" in track_info:
    rec = recommend_order(track_info["A"], track_info["B"])
    st.subheader("🧠 Recomendación de orden")
    st.write(f"**Sugerencia:** {rec} (considerando compatibilidad de tonalidad, diferencia de BPM y energía).")

# Mezcla
if file_a and file_b and st.button("Mezclar 🎶", type="primary"):
    with st.spinner("Procesando y mezclando pistas..."):
        try:
            y_a = track_info["A"]["y"]; sr_a = track_info["A"]["sr"]
            y_b = track_info["B"]["y"]; sr_b = track_info["B"]["sr"]

            # Unificar SR
            if sr_a != sr_b:
                y_b = librosa.resample(y_b, orig_sr=sr_b, target_sr=sr_a)
                sr_b = sr_a

            # Metadatos
            bpm_a = track_info["A"]["bpm"]; key_pc_a = track_info["A"]["pc"]
            bpm_b = track_info["B"]["bpm"]; key_pc_b = track_info["B"]["pc"]

            # Tonalidad objetivo: igualar a la de A si align_key
            target_key_pc = key_pc_a if align_key else None

            # Alinear a BPM objetivo + tono
            y_a_aligned = align_to_target(y_a, sr_a, bpm_a, key_pc_a, target_bpm, target_key_pc, align_tempo, align_key)
            y_b_aligned = align_to_target(y_b, sr_b, bpm_b, key_pc_b, target_bpm, target_key_pc, align_tempo, align_key)

            # Mezclar
            if mix_method == "Sincronización por Beats":
                mixed = beat_sync_crossfade(y_a_aligned, y_b_aligned, sr_a, a_mix_start, b_mix_start, int(overlap))
            else:
                mixed = time_crossfade(y_a_aligned, y_b_aligned, sr_a, a_mix_start, b_mix_start, overlap_seconds=overlap)

            # Recorte con extra
            extra_samples = int(extra_sec * sr_a)
            start = max(0, int(a_mix_start * sr_a) - extra_samples)
            end_candidate = int(b_mix_start * sr_a) + extra_samples
            end = min(len(mixed), end_candidate if end_candidate > start else len(mixed))
            final_mix = mixed[start:end] if end > start else mixed

            wav_bytes = write_wav_to_bytes(final_mix, sr_a)

            st.success("✅ ¡Mezcla completada!")
            st.audio(wav_bytes, format='audio/wav')
            st.download_button("⬇️ Descargar mezcla", data=wav_bytes, file_name="mezcla_dj_pro.wav", mime="audio/wav")

            with st.expander("Detalles de sesión"):
                st.json({
                    "SR": sr_a,
                    "BPM objetivo": target_bpm,
                    "BPM A / B": [round(bpm_a,2), round(bpm_b,2)],
                    "Tonalidad A / B": [track_info['A']['key'], track_info['B']['key']],
                    "Género A / B": [track_info['A'].get('top_genre'), track_info['B'].get('top_genre')],
                    "Método": mix_method
                })

        except Exception as e:
            st.error(f"Error durante la mezcla: {e}")
            st.exception(e)
else:
    st.info("👆 Sube dos audios y pulsa 'Mezclar 🎶' para crear tu mix.")
