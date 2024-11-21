import pyaudio
import wave
import tempfile
from groq import Groq
import pvporcupine
import numpy as np
from elevenlabs import generate, Voice, VoiceSettings, set_api_key
import sounddevice as sd
import soundfile as sf
import os
from openai import OpenAI
import json

# Initialize clients
set_api_key(ELEVEN_LABS_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class VoiceAssistant:
    def __init__(self):
        # Initialize voice settings
        self.voice_settings = VoiceSettings(
            stability=0.75,
            similarity_boost=0.75,
            style=0.6,
            use_speaker_boost=True
        )
        
        # Voice configurations
        self.voice_configs = {
            'normal': {
                'voice_id': 'ThT5KcBeYPX3keUQqHPh',  # Nicole - warm, friendly voice
                'model': 'eleven_multilingual_v2'
            }
        }

        # PyAudio configuration
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 512
        self.RECORD_SECONDS = 5

        # Initialize Porcupine
        self.porcupine = pvporcupine.create(
            access_key='JPEVmnznxg/tG181hMwyjvIAfwW+qvCKu6qpOcd6VOVQzfqF5yDnTg==',
            keywords=['Hey Saathi'],
            sensitivities=[0.5]
        )

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Create cache directory
        self.cache_dir = "voice_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # State
        self.is_running = False

    def _create_cache_key(self, text: str, message_type: str = 'normal') -> str:
        """Create a cache key for the audio file"""
        return f"{hash(text + message_type)}"

    def generate_speech(self, text: str, message_type: str = 'normal') -> str:
        """Generate speech using ElevenLabs API"""
        try:
            # Create cache path
            cache_key = self._create_cache_key(text, message_type)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.mp3")
            
            # Return cached audio if available
            if os.path.exists(cache_path):
                return cache_path
            
            # Get voice configuration
            voice_config = self.voice_configs.get(message_type, self.voice_configs['normal'])
            
            # Generate audio
            audio = generate(
                text=text,
                voice=voice_config['voice_id'],
                model=voice_config['model']
            )
            
            # Save audio to file
            with open(cache_path, 'wb') as f:
                f.write(audio)
            
            return cache_path
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None

    def play_audio(self, file_path: str):
        """Play audio file using sounddevice"""
        try:
            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Error playing audio: {str(e)}")

    def process_with_gpt4(self, text: str) -> str:
        """Process text with GPT-4 and return response"""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful and friendly assistant."},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error processing with GPT-4: {str(e)}")
            return "I apologize, but I'm having trouble processing your request at the moment."

    def record_and_transcribe(self):
        """Record audio and transcribe it using Groq"""
        frames = []
        
        # Open stream
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("Recording...")
        
        # Record audio
        for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        print("Finished recording")
        
        # Clean up stream
        stream.stop_stream()
        stream.close()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            with wave.open(temp_wav.name, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
            
            # Transcribe with Groq
            with open(temp_wav.name, 'rb') as audio_file:
                try:
                    transcription = groq_client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-1",
                        response_format="text",
                        language="en",
                        temperature=0.0
                    )
                    return transcription.strip()
                except Exception as e:
                    print(f"Error during transcription: {e}")
                    return None

    def run(self):
        """Main loop for the voice assistant"""
        self.is_running = True
        
        # Open stream for wake word detection
        wake_stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )
        
        try:
            print("Listening for wake word 'Hey Saathi'...")
            
            while self.is_running:
                # Read audio frame for wake word detection
                pcm = wake_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                # Check for wake word
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    print("Wake word detected! Listening for command...")
                    
                    # Record and transcribe command
                    transcription = self.record_and_transcribe()
                    
                    if transcription:
                        print(f"Transcription: {transcription}")
                        
                        # Process with GPT-4
                        response = self.process_with_gpt4(transcription)
                        print(f"GPT-4 Response: {response}")
                        
                        # Generate and play speech
                        audio_path = self.generate_speech(response)
                        if audio_path:
                            self.play_audio(audio_path)
                    
                    print("Listening for wake word 'Hey Saathi'...")
        
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            # Clean up
            wake_stream.stop_stream()
            wake_stream.close()
            self.audio.terminate()
            self.porcupine.delete()

def main():
    assistant = VoiceAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
