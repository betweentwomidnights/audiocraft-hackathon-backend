from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import torchaudio
import torch
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pydub import AudioSegment
import random
import json
from werkzeug.utils import secure_filename
import base64


app = Flask(__name__)
CORS(app)



@app.route('/generate', methods=['POST'])
def generate_audio():
    if request.json is None:
        return "No JSON received", 400

    audio_base64 = request.json.get('audioBase64')
    if audio_base64 is None:
        return "No audio received", 400

    print("Received audio:", audio_base64)

    # Get form data
    bpm = int(request.json.get('bpm', 75))
    prompt_duration = int(request.json.get('duration', 5))
    n_iterations = int(request.json.get('iterations', 7))
    output_duration_range = request.json.get('outputDurationRange', '20-30')
    min_duration, max_duration = map(int, output_duration_range.split('-'))

    # Get the base64-encoded audio and decode it
    audio_bytes = base64.b64decode(audio_base64)
    
    print(f"Received bpm: {bpm}, prompt_duration: {prompt_duration}, n_iterations: {n_iterations}")

    # Save the uploaded base64-decoded audio data temporarily
    filename = "temp_audio.wav"
    filepath = f"./tmp/{filename}"
    with open(filepath, "wb") as f:
        f.write(audio_bytes)

    # Load audio into a usable format
    song, sr = torchaudio.load(filepath)

    def peak_normalize(y, target_peak=0.9):
        return target_peak * (y / np.max(np.abs(y)))

    def rms_normalize(y, target_rms=0.05):
        return y * (target_rms / np.sqrt(np.mean(y**2)))

    def preprocess_audio(waveform):
        waveform_np = waveform.squeeze().numpy()
        processed_waveform_np = rms_normalize(peak_normalize(waveform_np))
        return torch.from_numpy(processed_waveform_np).unsqueeze(0)
    
    def calculate_duration():
        single_bar_duration = 4 * 60 / bpm
        bars = max(min_duration // single_bar_duration, 1)
        while single_bar_duration * bars < min_duration:
            bars += 1
        duration = single_bar_duration * bars
        while duration > max_duration and bars > 1:
            bars -= 1
            duration = single_bar_duration * bars
        return duration
    
    
    duration = calculate_duration()

    def create_slices(song, sr, slice_duration, num_slices=5):
        song_length = song.shape[-1] / sr
        slices = []
        for i in range(num_slices):
            random_start = random.choice(range(0, int((song_length - slice_duration) * sr), int(4 * 60 / bpm * sr)))
            slice_waveform = song[..., random_start:random_start + int(slice_duration * sr)]
            if len(slice_waveform.squeeze()) < int(slice_duration * sr):
                additional_samples_needed = int(slice_duration * sr) - len(slice_waveform.squeeze())
                slice_waveform = torch.cat([slice_waveform, song[..., :additional_samples_needed]], dim=-1)
            slices.append(slice_waveform)
        return slices
    
    slices = create_slices(song, sr, duration, num_slices=5)

    model_continue = MusicGen.get_pretrained('facebook/musicgen-small')
    model_continue.set_generation_params(duration=duration)

    all_audio_files = []

    for i in range(n_iterations):
        slice_idx = i % len(slices)
        prompt_waveform = slices[slice_idx][..., :int(prompt_duration * sr)]
        prompt_waveform = preprocess_audio(prompt_waveform)
        output = model_continue.generate_continuation(prompt_waveform, prompt_sample_rate=sr, progress=True)

        if len(output.size()) > 2:
            output = output.squeeze()
        
        filename = f'continue_{i}.wav'
        audio_write(filename, output.cpu(), model_continue.sample_rate, strategy="loudness", loudness_compressor=True)
        all_audio_files.append(filename)

        combined_audio = AudioSegment.empty()
        for filename in all_audio_files:
            combined_audio += AudioSegment.from_wav(filename)

    audio_stream = io.BytesIO()
    combined_audio.export(audio_stream, format='wav')
    audio_stream.seek(0)
    
    return send_file(
        audio_stream,
        as_attachment=True,
        download_name='combined_audio.wav',
        mimetype='audio/wav'
    )

if __name__ == '__main__':
    app.run(port=5000)
