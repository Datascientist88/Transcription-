from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv

app = Flask(__name__)
client = OpenAI()

def speech_to_text(audio_file):
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=f
        )
    return transcript

def get_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_data = request.files['audio']
    with open(audio_data.filename, "wb") as f:
        f.write(audio_data.read())
    transcript = speech_to_text(audio_data.filename)
    os.remove(audio_data.filename)
    return jsonify({'transcript': transcript})

@app.route('/response', methods=['GET'])
def response():
    prompt = request.args.get('prompt')
    response = get_response(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
