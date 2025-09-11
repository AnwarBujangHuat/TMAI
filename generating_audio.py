import os
import random
import librosa
import soundfile as sf
import numpy as np
from TTS.api import TTS
from pathlib import Path
import argparse

def prepare_reference_audio(input_dir, output_file="reference_voice.wav", duration_seconds=10):
    """
    Combine multiple recordings to create a reference voice sample for cloning.
    XTTS works best with 6-30 seconds of reference audio.
    """
    print("Preparing reference audio for voice cloning...")
    
    audio_files = list(Path(input_dir).glob("*.wav"))
    if len(audio_files) < 5:
        raise ValueError(f"Need at least 5 audio files, found {len(audio_files)}")
    
    # Select a few good quality samples
    selected_files = random.sample(audio_files, min(10, len(audio_files)))
    
    combined_audio = []
    target_sr = 22050
    
    for file_path in selected_files:
        audio, sr = librosa.load(file_path, sr=target_sr)
        # Trim silence and normalize
        audio, _ = librosa.effects.trim(audio, top_db=20)
        audio = librosa.util.normalize(audio)
        combined_audio.extend(audio)
        
        # Stop when we have enough audio
        if len(combined_audio) / target_sr >= duration_seconds:
            break
    
    # Trim to exact duration
    combined_audio = np.array(combined_audio)
    target_samples = int(duration_seconds * target_sr)
    combined_audio = combined_audio[:target_samples]
    
    # Save reference file
    sf.write(output_file, combined_audio, target_sr)
    print(f"Reference audio saved: {output_file} ({len(combined_audio)/target_sr:.1f}s)")
    return output_file

def generate_voice_variations(reference_audio_path, output_dir, num_samples=500):
    """
    Generate voice clones with variations using XTTS v2.
    """
    print("Loading XTTS model (this may take a while)...")
    
    # Initialize TTS with voice cloning model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Text variations for "hey tm"
    text_variations = [
        "hey tm",
        "Hey TM", 
        "hey T M",
        "Hey T M",
        "hey tea em",
        "Hey tea em",
        "hey t.m.",
        "Hey T.M.",
        "hey tim",  # phonetically similar
        "Hey Tim",
        "hey team",
        "Hey team",
    ]
    
    print(f"Generating {num_samples} voice clones...")
    
    for i in range(num_samples):
        try:
            # Select random text variation
            text = random.choice(text_variations)
            
            # Add slight variations to generation parameters
            temperature = random.uniform(0.75, 1.0)  # Controls randomness
            
            output_path = os.path.join(output_dir, f"hey_tm_clone_{i:04d}.wav")
            
            # Generate with voice cloning
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=reference_audio_path,
                language="en",
                # These parameters add variation
                **{"temperature": temperature}
            )
            
            # Apply post-processing for more variation
            audio, sr = librosa.load(output_path, sr=16000)
            audio = apply_audio_variations(audio, sr)
            sf.write(output_path, audio, sr)
            
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{num_samples} samples...")
                
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue
    
    print(f"Voice cloning complete! Generated samples in: {output_dir}")

def apply_audio_variations(audio, sr=16000):
    """
    Apply random variations to make the audio more diverse.
    """
    # Random pitch shift (-2 to +2 semitones)
    pitch_factor = random.uniform(-2, 2)
    if abs(pitch_factor) > 0.1:  # Only apply if significant
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor)
    
    # Random time stretch (0.9 to 1.1x speed)
    speed_factor = random.uniform(0.92, 1.08)
    if abs(speed_factor - 1.0) > 0.01:
        audio = librosa.effects.time_stretch(audio, rate=speed_factor)
    
    # Random volume adjustment
    volume_factor = random.uniform(0.7, 1.0)
    audio = audio * volume_factor
    
    # Add subtle background noise
    if random.random() > 0.7:  # 30% chance
        noise_level = random.uniform(0.001, 0.005)
        noise = np.random.normal(0, noise_level, audio.shape)
        audio = audio + noise
    
    # Ensure proper length (trim or pad to ~1 second)
    target_length = sr  # 1 second at 16kHz
    if len(audio) > target_length:
        # Random crop
        start_idx = random.randint(0, len(audio) - target_length)
        audio = audio[start_idx:start_idx + target_length]
    elif len(audio) < target_length:
        # Pad with silence
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    
    # Normalize
    audio = librosa.util.normalize(audio)
    
    return audio

def main():
    parser = argparse.ArgumentParser(description="Generate voice clones for wake word training")
    parser.add_argument("--input_dir", required=True, help="Directory with your original 300 recordings")
    parser.add_argument("--output_dir", default="generated_voice_clones", help="Output directory for generated samples")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of voice clones to generate")
    parser.add_argument("--reference_duration", type=int, default=10, help="Duration of reference audio in seconds")
    
    args = parser.parse_args()
    
    try:
        # Step 1: Prepare reference audio
        reference_path = prepare_reference_audio(
            args.input_dir, 
            "reference_voice.wav", 
            args.reference_duration
        )
        
        # Step 2: Generate voice clones
        generate_voice_variations(
            reference_path, 
            args.output_dir, 
            args.num_samples
        )
        
        print("\nVoice cloning pipeline complete!")
        print(f"Original samples: {len(list(Path(args.input_dir).glob('*.wav')))}")
        print(f"Generated samples: {len(list(Path(args.output_dir).glob('*.wav')))}")
        print(f"Total training data: {len(list(Path(args.input_dir).glob('*.wav'))) + len(list(Path(args.output_dir).glob('*.wav')))}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have TTS installed: pip install TTS")
        print("2. Ensure your input directory contains .wav files")
        print("3. Check that you have enough disk space")
        print("4. For GPU acceleration: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    main()