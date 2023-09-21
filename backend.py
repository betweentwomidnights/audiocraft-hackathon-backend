from flask import Flask, request, send_file
from flask_cors import CORS
import io
import torchaudio
import torch
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pydub import AudioSegment
import random
import base64
import os
from datetime import datetime
import logging
from waitress import serve

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

def conditional_cleanup(directory_path, size_limit_mb=100):
    total_size = 0

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        total_size += os.path.getsize(filepath) / (1024 * 1024)  # size in MB

    if total_size > size_limit_mb:
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            os.remove(filepath)
        logging.info(f"Cleared {directory_path} as it crossed {size_limit_mb} MB")

def preprocess_audio(waveform):
    def peak_normalize(y, target_peak=0.9):
        return target_peak * (y / np.max(np.abs(y)))

    def rms_normalize(y, target_rms=0.05):
        return y * (target_rms / np.sqrt(np.mean(y**2)))

    waveform_np = waveform.squeeze().numpy()
    processed_waveform_np = rms_normalize(peak_normalize(waveform_np))
    return torch.from_numpy(processed_waveform_np).unsqueeze(0)

def calculate_duration(bpm, min_duration, max_duration):
    single_bar_duration = 4 * 60 / bpm
    bars = max(min_duration // single_bar_duration, 1)
    while single_bar_duration * bars < min_duration:
        bars += 1
    duration = single_bar_duration * bars
    while duration > max_duration and bars > 1:
        bars -= 1
        duration = single_bar_duration * bars
    return duration

def create_slices(song, sr, slice_duration, num_slices=5):
    song_length = song.shape[-1] / sr
    slices = []
    first_slice_waveform = song[..., :int(slice_duration * sr)]
    slices.append(first_slice_waveform)

    for i in range(1, num_slices):
        random_start = random.choice(range(0, int((song_length - slice_duration) * sr), int(4 * 60 / 69.31 * sr)))
        slice_waveform = song[..., random_start:random_start + int(slice_duration * sr)]
        if len(slice_waveform.squeeze()) < int(slice_duration * sr):
            additional_samples_needed = int(slice_duration * sr) - len(slice_waveform.squeeze())
            slice_waveform = torch.cat([slice_waveform, song[..., :additional_samples_needed]], dim=-1)
        slices.append(slice_waveform)
    return slices

@app.route('/generate', methods=['POST'])
def generate_audio():
    conditional_cleanup('./tmp', 500)  # Clear if folder size > 500 MB
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    logging.info(f"Starting generation for {unique_id}")

    if request.json is None:
        return "No JSON received", 400

    audio_base64 = request.json.get('audioBase64')
    if audio_base64 is None:
        return "No audio received", 400

    combined_audio = AudioSegment.empty()
    all_audio_files = []
    
    bpm = int(request.json.get('bpm', 75))
    prompt_duration = int(request.json.get('duration', 5))
    n_iterations = int(request.json.get('iterations', 7))
    output_duration_range = request.json.get('outputDurationRange', '20-30')
    min_duration, max_duration = map(int, output_duration_range.split('-'))

    duration = calculate_duration(bpm, min_duration, max_duration)
    audio_bytes = base64.b64decode(audio_base64)
    filename = f"./tmp/temp_audio_{unique_id}.wav"

    with open(filename, "wb") as f:
        f.write(audio_bytes)

    song, sr = torchaudio.load(filename)
    slices = create_slices(song, sr, duration)
    model_continue = MusicGen.get_pretrained('facebook/musicgen-small')
    model_continue.set_generation_params(duration=duration)

    for i in range(n_iterations):
            logging.info(f"Starting iteration {i}")
            slice_idx = i % len(slices)
            
            try:
                prompt_waveform = preprocess_audio(slices[slice_idx][..., :int(prompt_duration * sr)])
                output = model_continue.generate_continuation(prompt_waveform, prompt_sample_rate=sr, progress=True)
                if len(output.size()) > 2:
                    output = output.squeeze()
                filename_with_extension = f"./tmp/continue_{unique_id}_{i}"
                audio_write(filename_with_extension, output.cpu(), model_continue.sample_rate, strategy="loudness", loudness_compressor=True)
                all_audio_files.append(filename_with_extension + '.wav')
            except Exception as e:
                logging.error(f"Error in iteration {i}: {e}")

    for filepath in all_audio_files:
        if os.path.exists(filepath):
            segment = AudioSegment.from_wav(filepath)
            combined_audio += segment

    audio_stream = io.BytesIO()
    combined_audio.export(audio_stream, format='wav')
    audio_stream.seek(0)

    return send_file(
        audio_stream,
        as_attachment=True,
        download_name=f'combined_audio_{unique_id}.wav',
        mimetype='audio/wav'
    )

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)

