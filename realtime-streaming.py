import pyaudio
import numpy as np
import wave
import io
import tempfile
import os
import uuid
from groq import Groq
from queue import Queue
import openai
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from playsound import playsound
import json
import time
from pathlib import Path
from dotenv import load_dotenv

class AudioConfig:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 16000
    RECORD_SECONDS = 5

class SimpleKnowledgeBase:
    """Knowledge base for elderly care assistant."""
    def __init__(self):
        self.knowledge_base = [
            {
                "topic": "communication",
                "guidelines": """
                - Speak clearly and avoid technical jargon
                - Use respectful and patient language
                - Maintain a warm and friendly tone
                - Address concerns with empathy
                - Repeat important information when necessary
                - Break down complex information into simple steps
                """
            },
            {
                "topic": "health",
                "guidelines": """
                - Encourage regular health check-ups
                - Provide clear medication reminders
                - Suggest gentle exercise routines
                - Offer nutrition advice
                - Address mobility concerns sensitively
                """
            },
            {
                "topic": "safety",
                "guidelines": """
                - Provide clear emergency instructions
                - Remind about home safety measures
                - Offer fall prevention tips
                - Suggest regular safety checks
                """
            }
        ]
        self.vectorizer = TfidfVectorizer()
        self.documents = [f"Topic: {item['topic']}\n{item['guidelines']}" for item in self.knowledge_base]
        self.vectors = self.vectorizer.fit_transform(self.documents)
    
    def get_relevant_context(self, query: str) -> str:
        """Get relevant context using TF-IDF similarity."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        most_similar_idx = similarities.argsort()[-2:][::-1]
        
        relevant_docs = [self.documents[idx] for idx in most_similar_idx]
        return "\n\n".join(relevant_docs)

class APIKeyValidator:
    @staticmethod
    def validate_groq_key(api_key: str) -> bool:
        try:
            client = Groq(api_key=api_key)
            # Try a minimal API call to verify the key
            client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model="mixtral-8x7b-32768",
                max_tokens=1
            )
            return True
        except Exception as e:
            print(f"Groq API Key Validation Error: {str(e)}")
            return False

    @staticmethod
    def validate_eleven_labs_key(api_key: str) -> bool:
        try:
            client = ElevenLabs(api_key=api_key)
            # Try to list voices to verify the key
            client.voices.get_all()
            return True
        except Exception as e:
            print(f"ElevenLabs API Key Validation Error: {str(e)}")
            return False

    @staticmethod
    def validate_openai_key(api_key: str) -> bool:
        try:
            client = openai.Client(api_key=api_key)
            # Try a minimal API call to verify the key
            client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4o", 
                max_tokens=1
            )
            return True
        except Exception as e:
            print(f"OpenAI API Key Validation Error: {str(e)}")
            return False

class ElderlyCareAssistant:
    def __init__(self, groq_api_key, openai_api_key, eleven_labs_api_key):
        # Validate API keys before proceeding
        if not self._validate_api_keys(groq_api_key, openai_api_key, eleven_labs_api_key):
            raise ValueError("API key validation failed. Please check your API keys.")
            
        self.groq_client = Groq(api_key=groq_api_key)
        self.openai_client = openai.Client(api_key=openai_api_key)
        self.eleven_labs_client = ElevenLabs(api_key=eleven_labs_api_key)
        self.knowledge_base = SimpleKnowledgeBase()
        self.audio = pyaudio.PyAudio()
        self.transcription_cache = []
        self._initialize_audio_device()

    def _validate_api_keys(self, groq_key, openai_key, eleven_labs_key) -> bool:
        """Validate all API keys before initializing the assistant."""
        validation_results = {
            'Groq': APIKeyValidator.validate_groq_key(groq_key),
            'OpenAI': APIKeyValidator.validate_openai_key(openai_key),
            'ElevenLabs': APIKeyValidator.validate_eleven_labs_key(eleven_labs_key)
        }
        
        # Print validation results
        print("\nAPI Key Validation Results:")
        for service, is_valid in validation_results.items():
            status = "✓ Valid" if is_valid else "✗ Invalid"
            print(f"{service}: {status}")
        
        return all(validation_results.values())

    def _initialize_audio_device(self):
        """Initialize and test audio device."""
        try:
            test_stream = self.audio.open(
                format=AudioConfig.FORMAT,
                channels=AudioConfig.CHANNELS,
                rate=AudioConfig.RATE,
                input=True,
                frames_per_buffer=AudioConfig.CHUNK
            )
            test_stream.close()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio device: {e}")

    def text_to_speech(self, text: str, max_retries=3) -> str:
        """Convert text to speech with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.eleven_labs_client.text_to_speech.convert(
                    voice_id="a0euEDMIIr9cUObJf0DX", 
                    output_format="mp3_22050_32",
                    text=text,
                    model_id="eleven_turbo_v2_5",
                    voice_settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=1.0,
                        style=0.3,
                        use_speaker_boost=True,
                    ),
                )
                
                save_file_path = f"response_{uuid.uuid4()}.mp3"
                with open(save_file_path, "wb") as f:
                    for chunk in response:
                        if chunk:
                            f.write(chunk)
                return save_file_path
                
            except Exception as e:
                if "invalid_api_key" in str(e):
                    print("Error: Invalid ElevenLabs API key. Please check your credentials.")
                    return None
                elif attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to convert text to speech after {max_retries} attempts: {e}")
                    return None

    def process_audio_chunk(self, audio_data: bytes) -> str:
        """Process audio chunk and return transcription."""
        temp_wav_path = None
        try:
            temp_wav_path = self._save_audio_chunk(audio_data)
            with open(temp_wav_path, 'rb') as audio_file:
                return self._transcribe_audio(audio_file)
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

    def _save_audio_chunk(self, audio_data: bytes) -> str:
        """Save audio data to temporary WAV file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            with wave.open(temp_wav.name, 'wb') as wf:
                wf.setnchannels(AudioConfig.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(AudioConfig.RATE)
                wf.writeframes(audio_data)
            return temp_wav.name

    def _transcribe_audio(self, audio_file) -> str:
        """Transcribe audio with fallback options."""
        models = ["distil-whisper-large-v3-en", "whisper-large-v3-turbo"]
        
        for model in models:
            try:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model=model,
                    response_format="text",
                    temperature=0.0
                )
                if transcription:
                    return transcription.strip()
            except Exception as e:
                if "model_not_found" in str(e) and model != models[-1]:
                    print(f"Model {model} not found, trying alternative...")
                    audio_file.seek(0)
                    continue
                elif model == models[-1]:
                    print(f"Transcription failed with model {model}: {e}")
                    if "invalid_api_key" in str(e):
                        raise RuntimeError("Invalid Groq API key. Please verify your credentials.")
                    raise RuntimeError(f"Transcription failed with all models: {e}")

    def get_assistant_response(self, transcription: str) -> str:
        """Get AI response using GPT-4."""
        context = self.knowledge_base.get_relevant_context(transcription)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(context)
                    },
                    {"role": "user", "content": transcription}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to get AI response: {e}")

    def _get_system_prompt(self, context: str) -> str:
        return f"""You are a kind, patient, and empathetic assistant specifically designed to help elderly users. 
        Always:
        - Speak clearly and avoid technical terms
        - Use a warm, respectful tone
        - Be patient and offer to repeat information
        - Break down complex information into simple steps
        - Address health and safety concerns with care
        - Provide emotional support when needed
        
        Relevant context for this interaction:
        {context}
        """

    def play_audio_response(self, audio_file_path: str):
        """Play the audio response file."""
        try:
            playsound(audio_file_path)
        except Exception as e:
            print(f"Error playing audio response: {e}")

    def run(self):
        """Main loop for the assistant."""
        print("\nElderly-Focused Assistant Ready!")
        print("Speak clearly and I'll respond with care and patience.")
        
        try:
            while True:
                input("\nPress Enter to start recording (or Ctrl+C to exit)...")
                stream = self.audio.open(
                    format=AudioConfig.FORMAT,
                    channels=AudioConfig.CHANNELS,
                    rate=AudioConfig.RATE,
                    input=True,
                    frames_per_buffer=AudioConfig.CHUNK
                )

                print("Listening with care...")
                frames = []
                
                try:
                    for _ in range(0, int(AudioConfig.RATE / AudioConfig.CHUNK * AudioConfig.RECORD_SECONDS)):
                        data = stream.read(AudioConfig.CHUNK, exception_on_overflow=False)
                        frames.append(data)

                    audio_data = b''.join(frames)
                    transcription = self.process_audio_chunk(audio_data)
                    
                    if transcription:
                        print(f"I heard: {transcription}")
                        self.transcription_cache.append(transcription)
                        
                        response_text = self.get_assistant_response(transcription)
                        print("Assistant Response:", response_text)
                        
                        audio_file_path = self.text_to_speech(response_text)
                        if audio_file_path:
                            print(f"Playing audio response...")
                            self.play_audio_response(audio_file_path)
                            # Optionally remove the audio file after playing
                            os.remove(audio_file_path)
                            print("Audio response complete.")

                except KeyboardInterrupt:
                    print("\nGently stopping the session...")
                    break
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print("Let's try again...")
                finally:
                    stream.stop_stream()
                    stream.close()
                    print("I'm ready to listen again...")

        finally:
            self.audio.terminate()
            print("Thank you for talking with me. Take care and stay safe!")

def main():
    load_dotenv()
    dotenv_path = Path('Code-and-Conquer')
    eleven_labs_api_key = os.getenv(ELEVEN_LABS_API_KEY)
    openai_api_key = os.getenv(OPENAI_API_KEY)
    groq_api_key = os.getenv(GROQ_API_KEY)
   
    try:
        print("\nInitializing Elderly Care Assistant...")
        print("Validating API keys...")
        assistant = ElderlyCareAssistant(groq_api_key, openai_api_key, eleven_labs_api_key)
        assistant.run()
    except ValueError as e:
        print(f"\nInitialization Error: {e}")
        print("Please check your API keys and try again.")
    except Exception as e:
        print(f"\nCritical error: {e}")
        print("Please check your API keys and try again.")

if __name__ == "__main__":
    main()