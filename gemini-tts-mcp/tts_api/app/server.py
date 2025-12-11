from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import texttospeech
import base64

app = FastAPI()

class TTSRequest(BaseModel):
    prompt: str
    text: str

@app.post("/tts")
def tts(req: TTSRequest):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(
        text=req.text,
        prompt=req.prompt
    )

    voice = texttospeech.VoiceSelectionParams(
        language_code="id-ID",
        name="Achernar",
        model_name="gemini-2.5-flash-tts"
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    audio_b64 = base64.b64encode(response.audio_content).decode()

    return {
        "audio_base64": audio_b64,
        "voice": "Achernar",
        "model": "gemini-2.5-flash-tts",
        "status": "success"
    }
