<p align="center">
  In this project, we explore various artificial neural network approaches to achieve high accuracy in a music genre classification task. Our method involves converting raw .wav audio input into an array of MFCC values.
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Request Feature</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About this Project

This project was conducted in collaboration with Professor Rebecca Lin (Feng Chia University, Taiwan) during my time on the Taiwan Experience Exchange Program (TEEP). Our results are based on the [GTZAN](http://marsyas.info/index.html) music genre dataset, which provides 10 human-classified genre folders: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock. Each genre folder contains 100 30-second audio clips of genre-specific songs in .wav format. We were able to achieve accuracies as high as 90%.


<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

## Prerequisite

Install [Python3](https://www.python.org/downloads/) on your computer.

Enter this into your computer's command line interface (terminal, control panel, etc.) to check the version:

  ```sh
  python --version
  ```

If the first number is not a 3, update to Python3.

## Setup

Here is an easy way to use our GitHub repository.

### Step 1: Clone the repository


Open the command line interface and run:
  ```sh
  git clone https://github.com/ericodle/GRU_Classifying_GTZAN.git
  ```

You have now downloaded the entire project, including all its sub-directories (folders) and files.
(We will avoid using Git commands.)

### Step 2: Navigate to the project directory
Find where your computer saved the project, then enter:

  ```sh
  cd /path/to/project/directory
  ```

If performed correctly, your command line interface should resemble

```
user@user:~/GRU_Classifying_GTZAN-main$
```

### Step 3: Create a virtual environment: 
Use a **virtual environment** so library versions on your computer match the versions used during development and testing.


```sh
python3 -m venv gtzan-env
```

A virtual environment named "gtzan-env" has been created. 
Enter the environment to do our work by using the following command:


```sh
source gtzan-env/bin/activate
```

When performed correctly, your command line interface prompt should look like 

```
(gtzan-env) user@user:~/GRU_Classifying_GTZAN-main$
```

### Step 3: Install requirements.txt

Avoid dependency hell by installing specific software versions known to work well together.

  ```sh
pip install -r requirements.txt
  ```

### Download GTZAN and extract MFCCs

 Training/testing music used for this project came from the GTZAN music genre dataset. 
 download and save the GTZAN dataset into the GRU_Classifying_GTZAN-main project folder.

#### Pre-process the GTZAN dataset into MFCCs

The GTZAN dataset is a collection of .wav files. We need to convert those into MFCCs for our analysis.

We can do this by using the MFCC_extraction.py script by passing arguments for the gtzan filepath, output path where you want to save the results, and the name you want to call the file

Here is an example:

```sh
# This script will extract MFCC's from each song clip.
python3 ./src/MFCC_extraction.py ./gtzan ./MFCCs gtzan_mfccs.json 
```
> Now, the resulting JSON file is saved in the "MFCCs" folder.

### Train a model from scratch

Run the following script to set up a new experiment with default settings.
Specify the type of neural network you want to use.
After training is complete, a train/validation curve will be saved in the project root directory.
Your final trained model state will also be saved for the next phase -- testing. 

   ```sh
   # Set up a new training run
   ./src/train_model.py
   ```
Note #1: Training requires a GPU to complete in a timely manner. You can either use your own hardware, or work in a Google Colab environment.
If you use a GPU, make sure you have CUDA and all related dependencies set up.

Note #2: Training a neural network involves trial and error with different model hyperparameters (hidden layers, learning rate, optimizer function, and more). Feel free to go into the train_model.py script and change these parameters as necessary. Default settings represent what worked best for us at the time of experimentation.

### Testing a trained model

Once you have a model trained from scratch on MFCCs extracted from the GTZAN music genre dataset, you should test how well it classifies music genres.
In our conference paper, we used a shuffled 80:10:10 split for training, train phase validation, and testing. Therefore, the music clip segments reserved for testing come from the same dataset but have never been seen by the model before.

  ```sh
  # Test a pre-trained model.
  ./src/test_model.py
  ```

Note: The entire MFCC JSON file is re-shuffled and split into 80:10:10 train/validation/test subsets each time the train_model.py and test_model.py scripts are run.

## Repository Files

- [ ] train_model.py

This script can be called to train a pre-defined neural network class on labeled MFCC data. Upon training completion, the user will be provided a graph of both training and validation following each train epoch. This graph can be useful in diagnosing neural network issues such as overfitting.

- [ ] test_model.py

This script can be called to test a trained neural network on labeled MFCC data. Once executed, a confusion matrix image will be generated to show the user how well the neural network was able to classify each song.

- [ ] models.py

This script defines the artificial neural network architectures used in our study. Classes for MLP, CNN, LSTM, BiLSTM, and GRU models are written using PyTorch, which we chose over Keras for its greater granular control.

- [ ] MFCC_extraction.py

This script extracts MFCCs from the GTZAN dataset music files and saves them in JSON format. The resulting file is about 640 MB in size, and contains an ordered list of 13 MFCC values per song segment. Genre classes are one-hot encoded with values 0 through 9 corresponding to the ten genres present.

- [ ] MFCC_primer.ipynb


This folder contains a collection of Jupyter Notebooks (linked to Google Colab) that were saved and uploaded to GitHub immediately after running. In their currently saved state, they serve as a record of the experimental runs on which we base our results. Users are welcome to play around with these scripts and try to beat our top test accuracy of 90.7%.

## Citing Our Research

Our research paper provides a comprehensive overview of the methodology, results, and insights derived from this repository. You can access the full paper by following this link: [Comparing Recurrent Neural Network Types in a Music Genre Classification Task: Gated Recurrent Unit Superiority Using the GTZAN Dataset](https://www.researchgate.net/publication/374698715_Comparing_Recurrent_Neural_Network_Types_in_a_Music_Genre_Classification_Task_Gated_Recurrent_Unit_Superiority_Using_the_GTZAN_Dataset)

<!-- LICENSE -->

## License
This project is open-source and is released under the [MIT License](LICENSE). Feel free to use and build upon our work while giving appropriate credit.


