import sys
import json
import os
import math
import librosa

# Ensure that the script can find the necessary modules.
sys.path.append('./')
sys.path.append('./src/')

# Define paths to the GTZAN dataset and the output directory for extracted features.
GTZAN_PATH = "./GTZAN_dataset/"
OUTPUT_PATH = "./MFCCs/"
OUTPUT_FILENAME = "mfcc_data.json"
OUTPUT_PATH = os.path.join(OUTPUT_PATH, OUTPUT_FILENAME)

# Constants for audio processing.
SAMPLE_RATE = 22050  # Standard sample rate for GTZAN audio data.
SONG_LENGTH = 30  # Duration of each song clip in seconds.
SAMPLE_COUNT = SAMPLE_RATE * SONG_LENGTH  # Total number of samples per clip.

def mfcc_to_json(gtzan_path, output_path, mfcc_count=13, n_fft=2048, hop_length=512, segs=10):
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio clips in the GTZAN dataset
    and saves the features along with labels into a JSON file.

    Args:
        gtzan_path (str): Path to the directory containing the GTZAN dataset.
        output_path (str): Path to the directory where the extracted MFCCs will be saved.
        mfcc_count (int): Number of MFCCs to extract.
        n_fft (int): Number of samples per FFT.
        hop_length (int): Number of samples between successive frames.
        segs (int): Number of segments to divide each audio clip into.

    Returns:
        None
    """
    # Initialize the data dictionary to store extracted features and labels.
    extracted_data = {
        "mapping": [],  # List to map numeric labels to genre names.
        "labels": [],   # List to store numeric labels for each audio clip.
        "mfcc": []      # List to store extracted MFCCs.
    }
    
    # Calculate the length of each segment.
    seg_length = int(SAMPLE_COUNT / segs)
    # Calculate the number of MFCCs per segment.
    mfccs_per_seg = math.ceil(seg_length / hop_length)

    # Loop through each genre folder in the GTZAN dataset.
    for i, (folder_path, folder_name, file_name) in enumerate(os.walk(gtzan_path)):
        if folder_path != gtzan_path:
            # Extract genre label from folder path.
            genre_label = folder_path.split("/")[-1]
            extracted_data["mapping"].append(genre_label)
            print("\nProcessing: {}".format(genre_label))

            # Iterate over each audio file in the genre folder.
            for song_clip in file_name:
                file_path = os.path.join(folder_path, song_clip)
                try:
                    # Load the audio file.
                    audio_sig, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    # Handle loading errors.
                    print(f"Error loading file {file_path}: {e}")
                    continue

                # Iterate over segments and calculate MFCCs.
                for k in range(segs):
                    # Define start and end indices for the current segment.
                    segment_start = seg_length * k
                    segment_end = segment_start + seg_length
                    try:
                        # Extract MFCCs for the current segment.
                        mfcc = librosa.feature.mfcc(y=audio_sig[segment_start:segment_end], sr=sr, n_mfcc=mfcc_count, n_fft=n_fft, hop_length=hop_length)
                        # Transpose the MFCC matrix.
                        mfcc = mfcc.T
                    except Exception as e:
                        # Handle MFCC extraction errors.
                        print(f"Error computing MFCCs for {file_path}, segment {k+1}: {e}")
                        continue

                    # Append MFCCs and label to the data dictionary.
                    if len(mfcc) == mfccs_per_seg:
                        extracted_data["mfcc"].append(mfcc.tolist())
                        extracted_data["labels"].append(i - 1)  # Subtract 1 to ensure zero-based indexing.
                        print("{}, segment:{}".format(file_path, k + 1))

    # Write the extracted data to a JSON file.
    try:
        with open(output_path, "w") as fp:
            json.dump(extracted_data, fp, indent=4)
            print(f"Successfully wrote data to {output_path}")
    except Exception as e:
        print(f"Error writing data to {output_path}: {e}")

# The resulting file will be approximately 640 MB.
# Ensure that GTZAN dataset is correctly placed in GTZAN_PATH before running the script.

# Classic Python safeguard to prevent accidental execution when importing as a module.
if __name__ == "__main__":
    mfcc_to_json(GTZAN_PATH, OUTPUT_PATH)

