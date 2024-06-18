# FSOAGAPI
This project is a Flask-based API that uses a pre-trained model from Stability AI to generate audio based on user-defined parameters. You can upload an initial audio file, set a prompt, and specify various generation parameters.

## Features

- **Custom Seed**: Provide a custom seed or let the API generate a random seed.
- **Audio Upload**: Upload an initial audio file to guide the generation process.
- **Parameter Configuration**: Set parameters like prompt, seconds total, steps, and noise level.

## Dependencies

The project uses the following Python packages:

- Flask
- torch
- torchaudio
- einops
- stable_audio_tools
- huggingface_hub
- mutagen

You can install them using `pip`:

```bash
pip install Flask torch torchaudio einops stable_audio_tools huggingface_hub mutagen
```

## Configuration

Set up the Hugging Face token to access the model.


## Usage

Start the Flask server with:

```bash
python app.py
```

The server will be running on `http://127.0.0.1:5000`.

### Endpoint

- **POST `/generate_audio`**

  **Request Parameters:**

  - `prompt` (string, default: "default prompt"): The prompt for audio generation.
  - `seconds_total` (integer, default: 60): The total duration for audio generation (max: 60).
  - `seed` (integer, optional): Custom seed for random number generation. If not provided, a random seed is generated.
  - `steps` (integer, default: 100): Number of steps for the generation process.
  - `cfg_scale` (float, default: 7): Configuration scale for the generation process.
  - `init_noise_level` (float, default: 10): Initial noise level for the generation.
  - `init_audio` (file, optional): Upload an initial audio file to guide the generation.
  - `init_audio_path` (string, optional): Path to an initial audio file. Default: "ARCS.wav".

  **Example Request:**

  ```http
  POST /generate_audio
  Content-Type: multipart/form-data

  prompt=distorted rhythmic filtered melodic sub-bass bassline loop
  seconds_total=60
  steps=100
  cfg_scale=7
  init_noise_level=10
  init_audio=@path/to/audio.wav
  ```

  **Response:**

  The server will return the generated audio file. You can download it directly.

## Example

Here's how to use the API with `curl`:

```bash
curl -X POST http://127.0.0.1:5000/generate_audio \
     -F "prompt=distorted rhythmic bassline" \
     -F "seconds_total=60" \
     -F "steps=100" \
     -F "cfg_scale=7" \
     -F "init_noise_level=10" \
     -F "init_audio=@path/to/your/audio.wav"
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

### How to Use:

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install Dependencies**: Install the required Python packages.
3. **Start the Server**: Run `python app.py` to start the Flask server.
4. **Make Requests**: Use `curl`, Postman, or any HTTP client to send POST requests to `http://127.0.0.1:5000/generate_audio`.

Feel free to modify the README to better fit your project’s details or add any additional instructions.
