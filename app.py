from flask import Flask, request, jsonify, send_file
import torch
import random
import torchaudio
from einops import rearrange
import stable_audio_tools
from stable_audio_tools.inference.generation import generate_diffusion_cond
from huggingface_hub.hf_api import HfFolder
import os
from mutagen.wave import WAVE
from mutagen.id3 import ID3, COMM
import io

app = Flask(__name__)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
HfFolder.save_token('REPLACE') # Use your own. https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/main

# Download model
model, model_config = stable_audio_tools.get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"] * 2
model = model.to(device)

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    prompt = request.form.get('prompt', "default prompt")
    seconds_total = int(request.form.get('seconds_total', 60))
    if seconds_total > 60:
        return jsonify({"error": "seconds_total cannot exceed 60"}), 400
    
    seed = request.form.get('seed')
    seed = int(seed) if seed else random.randint(0, 2**32 - 1)
    
    steps = int(request.form.get('steps', 100))
    cfg_scale = float(request.form.get('cfg_scale', 7))
    init_noise_level = float(request.form.get('init_noise_level', 10))
    
    # Check if a file is uploaded
    if 'init_audio' in request.files:
        init_audio_file = request.files['init_audio']
        init_audio_waveform, init_audio_sample_rate = torchaudio.load(io.BytesIO(init_audio_file.read()))
    else:
        init_audio_path = request.form.get('init_audio_path', "ARCS")
        init_audio_waveform, init_audio_sample_rate = torchaudio.load(init_audio_path + ".wav")
    
    # Ensure the sample rate matches
    if init_audio_sample_rate != sample_rate:
        init_audio_waveform = torchaudio.transforms.Resample(orig_freq=init_audio_sample_rate, new_freq=sample_rate)(init_audio_waveform)
    
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": seconds_total
    }]
    
    # Generate stereo audio
    output = generate_diffusion_cond(
        model,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        init_audio=(sample_rate, init_audio_waveform),
        init_noise_level=init_noise_level,
        device=device
    )
    
    # Save output audio to a file
    output_file_path = f'output_{seed}.wav'
    torchaudio.save(output_file_path, output.cpu(), sample_rate)
    
    return send_file(output_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
