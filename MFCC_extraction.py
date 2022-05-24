# Import required packages. The only one that may require installation is librosa.
import json
import os
import math
import librosa

# Ensure that your GTZAN dataset and output paths are correct.
# Also ensure that your copy of the GTZAN dataset is correctly structed by genre folder.
GTZAN_PATH = "your_gtzan_path_here"
OUTPUT_PATH = "your_output_path_here"

# We need to know some basic information about each .wav file.
SAMPLE_RATE = 22050 # This is the sample rate used in GTZAN audio.
SONG_LENGTH = 30 # Unit: seconds
SAMPLE_COUNT = SAMPLE_RATE * SONG_LENGTH

# We herein define the key function that extracts MFCCs from each song clip and saves it to a json file.
# The "segs" argument represents how many sub-clip segments of equal duration each audio clip should be divided into.
# Other argument values are regarded as conventional for this field.
def mfcc_to_json(gtzan_path, output_path, mfcc_count=13, n_fft=2048, hop_length=512, segs=10):

    # This variable defines the hierarchical structure of how our MFCC extracts will be organized.
    extracted_data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }
    
    seg_length = int(SAMPLE_COUNT / segs)
    mfccs_per_seg = math.ceil(seg_length / hop_length)

    # GTZAN is a big folder holding 10 smaller folders, each containing 100 song clips of the same genre.
    # Here we work through one genre folder at a time using the enumerate(os.walk()) function.
    # This process should keep our genre labels, song files, and MFCC values all in hierarchical order.
    for i, (folder_path, folder_name, file_name) in enumerate(os.walk(gtzan_path)):

        # This "IF" statement makes sure we are looking at one of the genre folders, not the main GTZAN folder.
	# If
        if folder_path is not gtzan_path:

            # save genre label (i.e., sub-folder name) in the mapping
            genre_label = folder_path.split("/")[-1]
            extracted_data["mapping"].append(genre_label)
            print("\nProcessing: {}".format(genre_label))

            # process all audio files in genre sub-dir
            for f in file_name:

		# load audio file
                file_path = os.path.join(folder_path, f)
                audio_sig, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(segs):

                    # We must segment each audio clip by defining the beginning and end.
                    GO = seg_length * d
                    STOP = GO + seg_length

                    # Here, we tell librosa to do the actual MFCC calculations.
                    mfcc = librosa.feature.mfcc(audio_sig[GO:STOP], sr, n_mfcc=mfcc_count, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # This "IF" statement will make sure that the MFCC extract set for a segment is the appropriate size.
		    # Possible problems include a song clip not exactly 30 seconds or a file in GTZAN that has become corrupted.
		    # We don't want to break the whole process for just one error in such a large dataset.
		    # S0, we simply omit it and move on.
                    if len(mfcc) == mfccs_per_seg:
                        extracted_data["mfcc"].append(mfcc.tolist())
                        extracted_data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))
		    
		

    # Finally, we DUMP the labeled mfcc extract data to a file at out specified output path.
    with open(output_path, "w") as fp:
        json.dump(extracted_data, fp, indent=4)
        
# The resulting file will be about 640 MB.
# If everything worked, the JSON file should start with the lines:
# Note that "pop" = label 0; "classical"= label 1; "jazz" = label 2....etc.
#{
#    "mapping": [
#        "pop",
#        "classical",
#        "jazz",
#        "hiphop",
#        "reggae",
#        "disco",
#        "metal",
#        "country",
#        "blues",
#        "rock"
#    ],
#    "labels": [

# Classic Python failsafe code that protects users from accidentally calling a script.
if __name__ == "__main__":
    mfcc_to_json(GTZAN_PATH, OUTPUT_PATH)
