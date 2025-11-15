#!/usr/bin/env python3
"""
Process groove dataset with audio files and beat annotations.
Chunks audio into 30-second segments and uploads to HuggingFace.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import soundfile as sf
import tempfile
from datasets import Dataset, DatasetDict, Audio, Features, Sequence, Value
from huggingface_hub import login
import argparse
from tqdm import tqdm


def parse_beat_file(beat_file_path: Path) -> List[Dict[str, float]]:
    """Parse a .beats file and return list of events."""
    events = []
    with open(beat_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    try:
                        timestamp = (1000 * float(parts[0])) / 1000
                        beat = int(parts[1])
                        events.append({
                            'timestamp': float(timestamp),  # Ensure float
                            'beat': int(beat)  # Ensure int
                        })
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Skipping invalid line in {beat_file_path}: {line}")
                        continue
    return events


def get_audio_info(audio_path: Path) -> Tuple[int, int]:
    """Get audio duration and sample rate."""
    info = sf.info(audio_path)
    return info.duration, info.samplerate


def chunk_audio_and_events(
    audio_path: Path,
    events: List[Dict[str, float]],
    chunk_duration: float = 30.0,
    output_dir: Optional[Path] = None
) -> List[Dict]:
    """
    Split audio and events into chunks of specified duration.
    Returns list of examples with audio chunks and adjusted events.
    """
    # Load audio
    audio_data, sample_rate = sf.read(audio_path)
    
    # Ensure audio is 1D (mono) - if stereo, convert to mono
    if len(audio_data.shape) > 1:
        # Average channels to convert to mono
        audio_data = np.mean(audio_data, axis=1)
    
    # Ensure audio_data is float32 and 1D
    audio_data = audio_data.astype(np.float32)
    if audio_data.ndim != 1:
        audio_data = audio_data.flatten()
    
    duration = len(audio_data) / sample_rate
    
    # Calculate number of chunks
    num_chunks = int(np.ceil(duration / chunk_duration))
    # Ensure output directory exists (use a temp directory if not provided)
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="groove_chunks_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunks = []
    for chunk_idx in range(num_chunks):
        start_time = chunk_idx * chunk_duration
        end_time = min((chunk_idx + 1) * chunk_duration, duration)
        
        # Extract audio chunk
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio_chunk = audio_data[start_sample:end_sample]
        
        # Ensure chunk is 1D and float32
        audio_chunk = audio_chunk.astype(np.float32)
        
        # Write the audio chunk to a temporary WAV file
        chunk_filename = f"{audio_path.stem}_chunk{chunk_idx:04d}.wav"
        chunk_path = output_dir / chunk_filename
        # soundfile expects shape (num_samples,) for mono float32
        sf.write(str(chunk_path), audio_chunk, int(sample_rate), subtype='PCM_16')
        
        # Filter and adjust events for this chunk
        chunk_events = []
        for event in events:
            if isinstance(event, dict) and 'timestamp' in event and 'beat' in event:
                if start_time <= event['timestamp'] < end_time:
                    # Ensure proper types for each field
                    chunk_events.append({
                        'timestamp': float(event['timestamp'] - start_time),  # Adjust relative to chunk start
                        'beat': int(event['beat'])
                    })
        
        # Ensure events is always a list, even if empty
        if not isinstance(chunk_events, list):
            chunk_events = []
        
        chunks.append({
            # Point to the chunk file path instead of embedding raw audio
            'audio': str(chunk_path),
            'events': chunk_events,
            'original_file': str(audio_path.name),
            'chunk_idx': chunk_idx,
            'start_time': float(start_time),
            'end_time': float(end_time)
        })
    
    return chunks


def match_audio_to_annotation(audio_path: Path, annotations_dir: Path) -> Optional[Path]:
    """
    Match an audio file to its corresponding annotation file.
    """
    # Parse audio filename
    audio_name = audio_path.stem  # Remove .wav extension
    
    # Get drummer and session info from path
    session_dir = audio_path.parent.name  # e.g., 'session1' or 'eval_session'
    drummer_dir = audio_path.parent.parent.name  # e.g., 'drummer1'
    
    # Parse the audio filename to extract components
    # Format: NUMBER_STYLE_TEMPO_beat_TIMESIG.wav
    # e.g., 1_funk-groove1_138_beat_4-4.wav
    parts = audio_name.split('_')
    if len(parts) >= 5:
        number = parts[0]
        style = parts[1]
        tempo = parts[2]
        time_sig = parts[4] if len(parts) > 4 else '4-4'
        
        # Construct annotation filename
        # Format: drummerN_sessionN_NUMBER_STYLE_TEMPO_beat_TIMESIG.beats
        annotation_name = f"{drummer_dir}_{session_dir}_{number}_{style}_{tempo}_beat_{time_sig}.beats"
        annotation_path = annotations_dir / annotation_name
        
        if annotation_path.exists():
            return annotation_path
        
        # Try alternative naming patterns
        # Sometimes the style name might be different
        for file in annotations_dir.glob(f"{drummer_dir}_{session_dir}_{number}_*_{tempo}_beat_{time_sig}.beats"):
            return file
    
    return None


def process_dataset(
    groove_dir: Path,
    annotations_dir: Path,
    chunk_duration: float = 30.0
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process entire dataset and return train and test examples.
    """
    train_examples = []
    test_examples = []
    # Create a temporary root directory to store all chunked audio files for this run
    temp_root_dir = Path(tempfile.mkdtemp(prefix="groove_chunks_"))
    
    # Iterate through all drummer directories
    for drummer_dir in sorted(groove_dir.iterdir()):
        if not drummer_dir.is_dir() or not drummer_dir.name.startswith('drummer'):
            continue
        
        print(f"Processing {drummer_dir.name}...")
        
        # Iterate through session directories
        for session_dir in sorted(drummer_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            
            is_eval = 'eval' in session_dir.name.lower()
            
            # Process all WAV files in session
            wav_files = list(session_dir.glob('*.wav'))
            
            for audio_path in tqdm(wav_files, desc=f"  {session_dir.name}"):
                # Find matching annotation file
                annotation_path = match_audio_to_annotation(audio_path, annotations_dir)
                
                if annotation_path is None:
                    print(f"    Warning: No annotation found for {audio_path}")
                    continue
                
                try:
                    # Parse beat annotations
                    events = parse_beat_file(annotation_path)
                    
                    # Validate events structure
                    if not isinstance(events, list):
                        print(f"    Warning: Events is not a list for {audio_path}")
                        continue
                    
                    # Check first few events
                    for i, event in enumerate(events[:3]):
                        if not isinstance(event, dict):
                            print(f"    Warning: Event {i} is not a dict: {type(event)} for {audio_path}")
                            events = []  # Reset to empty list if malformed
                            break
                        if 'timestamp' not in event or 'beat' not in event:
                            print(f"    Warning: Event {i} missing fields: {event.keys()} for {audio_path}")
                            events = []  # Reset to empty list if malformed
                            break
                    
                    # Chunk audio and events
                    chunks = chunk_audio_and_events(
                        audio_path,
                        events,
                        chunk_duration,
                        output_dir=temp_root_dir
                    )
                    
                    # Add metadata
                    for chunk in chunks:
                        chunk['drummer'] = drummer_dir.name
                        chunk['session'] = session_dir.name
                        chunk['is_eval'] = is_eval
                        
                        # Final validation of chunk structure
                        if not isinstance(chunk.get('events'), list):
                            print(f"    Warning: Chunk events is not a list for {audio_path}")
                            chunk['events'] = []
                    
                    # Add to appropriate split
                    if is_eval:
                        test_examples.extend(chunks)
                    else:
                        train_examples.extend(chunks)
                        
                except Exception as e:
                    print(f"    Error processing {audio_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Final validation of all examples
    print(f"\nValidating {len(train_examples)} train examples...")
    for i, ex in enumerate(train_examples[:5]):  # Check first 5
        if 'events' in ex and ex['events']:
            print(f"  Example {i}: {len(ex['events'])} events, first event: {ex['events'][0]}")
    
    print(f"Validating {len(test_examples)} test examples...")
    for i, ex in enumerate(test_examples[:5]):  # Check first 5
        if 'events' in ex and ex['events']:
            print(f"  Example {i}: {len(ex['events'])} events, first event: {ex['events'][0]}")
    
    return train_examples, test_examples


def create_huggingface_dataset(
    train_examples: List[Dict],
    test_examples: List[Dict]
) -> DatasetDict:
    """
    Create HuggingFace dataset from examples.
    """
    # Validate and clean examples
    def validate_example(example):
        """Ensure example has correct structure."""
        # Validate events field
        if 'events' not in example:
            example['events'] = []
        elif not isinstance(example['events'], list):
            example['events'] = []
        else:
            # Ensure all events are dicts with correct fields
            clean_events = []
            for event in example['events']:
                if isinstance(event, dict) and 'timestamp' in event and 'beat' in event:
                    clean_events.append({
                        'timestamp': float(event['timestamp']),
                        'beat': int(event['beat'])
                    })
            example['events'] = clean_events
        
        # Ensure other fields have correct types
        example['chunk_idx'] = int(example.get('chunk_idx', 0))
        example['start_time'] = float(example.get('start_time', 0.0))
        example['end_time'] = float(example.get('end_time', 30.0))
        example['is_eval'] = bool(example.get('is_eval', False))
        
        return example
    
    # Validate all examples
    train_examples = [validate_example(ex) for ex in train_examples]
    test_examples = [validate_example(ex) for ex in test_examples]
    
    # Define features - don't specify sampling_rate to let it be inferred from data
    features = Features({
        'audio': Audio(),  # Let sampling rate be inferred from the data
        'events': Sequence({
            'timestamp': Value('float32'),
            'beat': Value('int32')
        }),
        'original_file': Value('string'),
        'chunk_idx': Value('int32'),
        'start_time': Value('float32'),
        'end_time': Value('float32'),
        'drummer': Value('string'),
        'session': Value('string'),
        'is_eval': Value('bool')
    })
    
    # Create datasets
    print(f"Creating train dataset with {len(train_examples)} examples...")
    try:
        train_dataset = Dataset.from_list(train_examples)
    except Exception as e:
        print(f"Error creating train dataset: {e}")
        print("Debugging first train example:")
        if train_examples:
            ex = train_examples[0]
            print(f"  Keys: {ex.keys()}")
            print(f"  Events type: {type(ex.get('events'))}")
            if 'events' in ex and ex['events']:
                print(f"  First event: {ex['events'][0]}")
        raise
    
    print(f"Creating test dataset with {len(test_examples)} examples...")
    try:
        test_dataset = Dataset.from_list(test_examples)
    except Exception as e:
        print(f"Error creating test dataset: {e}")
        print("Debugging first test example:")
        if test_examples:
            ex = test_examples[0]
            print(f"  Keys: {ex.keys()}")
            print(f"  Events type: {type(ex.get('events'))}")
            if 'events' in ex and ex['events']:
                print(f"  First event: {ex['events'][0]}")
        raise

    train_dataset = train_dataset.cast_column('audio', Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description='Process groove dataset and upload to HuggingFace')
    parser.add_argument('--groove-dir', type=str, default='groove',
                        help='Path to groove directory with audio files')
    parser.add_argument('--annotations-dir', type=str, default='annotations',
                        help='Path to annotations directory with .beats files')
    parser.add_argument('--chunk-duration', type=float, default=30.0,
                        help='Maximum duration of audio chunks in seconds')
    parser.add_argument('--repo-name', type=str, required=True,
                        help='HuggingFace repository name (e.g., username/dataset-name)')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace API token (optional, can use login)')
    parser.add_argument('--private', action='store_true',
                        help='Make the repository private')
    parser.add_argument('--local-save', type=str, default=None,
                        help='Save dataset locally to this path (optional)')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    groove_dir = Path(args.groove_dir)
    annotations_dir = Path(args.annotations_dir)
    
    # Validate directories exist
    if not groove_dir.exists():
        raise ValueError(f"Groove directory not found: {groove_dir}")
    if not annotations_dir.exists():
        raise ValueError(f"Annotations directory not found: {annotations_dir}")
    
    # Login to HuggingFace if token provided
    if args.token:
        login(token=args.token)
    
    # Process dataset
    print("Processing dataset...")
    train_examples, test_examples = process_dataset(
        groove_dir,
        annotations_dir,
        args.chunk_duration
    )
    
    print(f"Found {len(train_examples)} training examples")
    print(f"Found {len(test_examples)} test (eval) examples")
    
    if len(train_examples) == 0 and len(test_examples) == 0:
        raise ValueError("No examples found! Check your directory structure and file naming.")
    
    # Create HuggingFace dataset
    print("Creating HuggingFace dataset...")
    dataset = create_huggingface_dataset(train_examples, test_examples)
    
    # Save locally if requested
    if args.local_save:
        print(f"Saving dataset locally to {args.local_save}...")
        dataset.save_to_disk(args.local_save)
    
    # Upload to HuggingFace
    print(f"Uploading to HuggingFace repository: {args.repo_name}")
    dataset.push_to_hub(
        args.repo_name,
        private=args.private
    )
    
    print("Done! Dataset uploaded successfully.")
    print(f"You can access it at: https://huggingface.co/datasets/{args.repo_name}")


if __name__ == "__main__":
    main()