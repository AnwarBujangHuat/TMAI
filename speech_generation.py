import asyncio
import edge_tts
from pydub import AudioSegment
import os
import numpy as np
import tempfile

async def generate_edge_tts_accents(base_text="HEY TM", output_dir="hey_tm_accents"):
    """Generate HEY TM using Microsoft Edge TTS with different voices and intonations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Various English voices with different accents
    voices = [
        {"name": "US_Female_1", "voice": "en-US-JennyNeural", "desc": "US Female (Jenny)"},
        {"name": "US_Male_1", "voice": "en-US-GuyNeural", "desc": "US Male (Guy)"},
        {"name": "UK_Female_1", "voice": "en-GB-SoniaNeural", "desc": "UK Female (Sonia)"},
        {"name": "UK_Male_1", "voice": "en-GB-RyanNeural", "desc": "UK Male (Ryan)"},
        {"name": "AU_Female_1", "voice": "en-AU-NatashaNeural", "desc": "Australian Female"},
        {"name": "AU_Male_1", "voice": "en-AU-WilliamNeural", "desc": "Australian Male"},
        {"name": "CA_Female_1", "voice": "en-CA-ClaraNeural", "desc": "Canadian Female"},
        {"name": "IN_Female_1", "voice": "en-IN-NeerjaNeural", "desc": "Indian English Female"},
        {"name": "ZA_Female_1", "voice": "en-ZA-LeahNeural", "desc": "South African Female"},
    ]
    
    # Different intonation patterns and variations
    intonation_variations = [
        {
            "name": "casual",
            "text": "Hey TM",
            "ssml_wrapper": lambda text: f'<speak><prosody rate="medium" pitch="medium">{text}</prosody></speak>',
            "desc": "Casual greeting"
        },
        {
            "name": "excited",
            "text": "HEEEEY TM!",
            "ssml_wrapper": lambda text: f'<speak><prosody rate="fast" pitch="high">{text}</prosody></speak>',
            "desc": "Excited/enthusiastic"
        },
        {
            "name": "elongated",
            "text": "Heyyyy TM",
            "ssml_wrapper": lambda text: f'<speak><prosody rate="slow" pitch="medium">{text}</prosody></speak>',
            "desc": "Drawn out/elongated"
        },
        {
            "name": "questioning",
            "text": "Hey TM?",
            "ssml_wrapper": lambda text: f'<speak><prosody pitch="high">{text}</prosody></speak>',
            "desc": "Questioning tone"
        },
        {
            "name": "whisper",
            "text": "Hey TM",
            "ssml_wrapper": lambda text: f'<speak><prosody volume="soft" rate="slow">{text}</prosody></speak>',
            "desc": "Whispered"
        },
        {
            "name": "emphatic",
            "text": "HEY! TM!",
            "ssml_wrapper": lambda text: f'<speak><prosody pitch="high" volume="loud"><emphasis level="strong">{text}</emphasis></prosody></speak>',
            "desc": "Emphatic/commanding"
        },
        {
            "name": "friendly",
            "text": "Hey there, TM!",
            "ssml_wrapper": lambda text: f'<speak><prosody rate="medium" pitch="+10%">{text}</prosody></speak>',
            "desc": "Friendly greeting"
        },
        {
            "name": "sing_song",
            "text": "Hey TM",
            "ssml_wrapper": lambda text: f'<speak><prosody pitch="high" contour="(20%,+20Hz) (40%,+30Hz) (60%,-10Hz) (80%,+5Hz)">{text}</prosody></speak>',
            "desc": "Sing-song pattern"
        },
        {
            "name": "urgent",
            "text": "Hey! TM!",
            "ssml_wrapper": lambda text: f'<speak><prosody rate="fast" pitch="high" volume="loud">{text}</prosody></speak>',
            "desc": "Urgent call"
        },
        {
            "name": "calm",
            "text": "Hey TM",
            "ssml_wrapper": lambda text: f'<speak><prosody rate="slow" pitch="low" volume="medium">{text}</prosody></speak>',
            "desc": "Calm/relaxed"
        }
    ]
    
    successful_generations = []
    
    print(f"Generating human-like 'HEY TM' variations using Microsoft Edge TTS...")
    print("=" * 80)
    
    for voice_config in voices:
        voice_name = voice_config["name"]
        voice = voice_config["voice"]
        voice_desc = voice_config["desc"]
        
        print(f"\nProcessing voice: {voice_desc}")
        print("-" * 40)
        
        for variation in intonation_variations:
            var_name = variation["name"]
            text = variation["text"]
            ssml_text = variation["ssml_wrapper"](text)
            var_desc = variation["desc"]
            
            print(f"  Generating: {var_desc}")
            
            try:
                # Create TTS with SSML for intonation control
                tts = edge_tts.Communicate(ssml_text, voice)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    await tts.save(temp_file.name)
                    temp_mp3 = temp_file.name
                
                # File naming
                file_base = f"hey_tm_{voice_name.lower()}_{var_name}"
                raw_wav = os.path.join(output_dir, f"{file_base}_raw.wav")
                processed_wav = os.path.join(output_dir, f"{file_base}_processed.wav")
                
                # Convert MP3 to WAV
                audio = AudioSegment.from_file(temp_mp3)
                audio.export(raw_wav, format="wav")
                
                # Process to target duration
                if process_to_target_duration(raw_wav, processed_wav, target_duration=1.5):
                    successful_generations.append({
                        'file': processed_wav,
                        'accent': voice_desc,
                        'variation': var_desc,
                        'name': f"{voice_name}_{var_name}",
                        'text_used': text
                    })
                    print(f"    ✓ Success: {os.path.basename(processed_wav)}")
                else:
                    print(f"    ✗ Failed processing")
                
                # Cleanup temp file
                os.unlink(temp_mp3)
                
            except Exception as e:
                print(f"    ✗ Error: {str(e)}")

    return successful_generations

def process_to_target_duration(input_file, output_file, target_duration=1.5, target_sample_rate=16000):
    """Process audio to target duration with improved speech detection."""
    try:
        audio = AudioSegment.from_wav(input_file)
        audio = audio.set_frame_rate(target_sample_rate).set_channels(1)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        target_length = int(target_duration * target_sample_rate)
        
        # Improved energy-based speech detection
        # Use shorter window for better precision
        window_size = int(0.025 * target_sample_rate)  # 25ms window
        energy = np.convolve(samples**2, np.ones(window_size)/window_size, mode='same')
        
        # Dynamic threshold based on audio characteristics
        max_energy = np.max(energy)
        mean_energy = np.mean(energy)
        threshold = max(mean_energy * 2, max_energy * 0.05)
        
        speech_indices = np.where(energy > threshold)[0]
        
        if len(speech_indices) == 0:
            print(f"    Warning: No speech detected")
            return False
        
        # Add padding around speech
        padding_samples = int(0.1 * target_sample_rate)  # 100ms padding
        speech_start = max(0, speech_indices[0] - padding_samples)
        speech_end = min(len(samples), speech_indices[-1] + padding_samples)
        speech_samples = samples[speech_start:speech_end]
        
        # Handle different duration scenarios
        if len(speech_samples) >= target_length:
            # If longer than target, center the crop
            excess = len(speech_samples) - target_length
            start = excess // 2
            final_samples = speech_samples[start:start + target_length]
        else:
            # If shorter than target, pad with silence
            padding = target_length - len(speech_samples)
            left_pad = padding // 2
            right_pad = padding - left_pad
            final_samples = np.concatenate([
                np.zeros(left_pad),
                speech_samples,
                np.zeros(right_pad)
            ])
        
        # Apply gentle fade in/out to avoid clicks
        fade_samples = int(0.01 * target_sample_rate)  # 10ms fade
        final_samples[:fade_samples] *= np.linspace(0, 1, fade_samples)
        final_samples[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Convert back to audio
        final_audio = AudioSegment(
            (final_samples * 32767).astype(np.int16).tobytes(),
            frame_rate=target_sample_rate,
            sample_width=2,
            channels=1
        )
        
        final_audio.export(output_file, format="wav")
        
        # Calculate audio statistics
        rms = np.sqrt(np.mean(final_samples ** 2))
        duration = len(final_samples) / target_sample_rate
        
        print(f"    Duration: {duration:.2f}s, RMS: {rms:.4f}")
        
        return rms > 0.0005  # Lower threshold for whispered audio
        
    except Exception as e:
        print(f"    Error processing audio: {str(e)}")
        return False

async def main():
    print("Human-like 'HEY TM' Generator with Intonations")
    print("=" * 50)
    print("Generating multiple intonation patterns:")
    print("• Casual greeting")
    print("• Excited/enthusiastic") 
    print("• Drawn out/elongated")
    print("• Questioning tone")
    print("• Whispered")
    print("• Emphatic/commanding")
    print("• Friendly greeting")
    print("• Sing-song pattern")
    print("• Urgent call")
    print("• Calm/relaxed")
    print("=" * 50)
    
    try:
        results = await generate_edge_tts_accents()
        
        print("\n" + "=" * 80)
        print("GENERATION SUMMARY")
        print("=" * 80)
        
        if results:
            print(f"Successfully generated {len(results)} audio variants!")
            print(f"Files saved in: hey_tm_accents/")
            
            # Group by voice for better readability
            voices_summary = {}
            for result in results:
                voice = result['accent']
                if voice not in voices_summary:
                    voices_summary[voice] = []
                voices_summary[voice].append(result)
            
            print(f"\nBreakdown by voice:")
            for voice, voice_results in voices_summary.items():
                print(f"  {voice}: {len(voice_results)} variations")
                for res in voice_results:
                    print(f"    - {res['variation']} ({os.path.basename(res['file'])})")
            
            print(f"\n✓ All files ending with '_processed.wav' are ready for training!")
            print(f"✓ Target duration: 1.5 seconds per file")
            print(f"✓ Sample rate: 16kHz mono")
            
        else:
            print("❌ No files generated successfully.")
            print("Please check your internet connection and try again.")
            
    except ImportError as e:
        print("❌ Missing dependencies!")
        print("Install with: pip install edge-tts pydub numpy")
        print(f"Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())