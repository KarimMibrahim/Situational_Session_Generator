# Situational_Session_Generator

This is the code for the paper 'Audio Autotagging as Proxy for Contextual MusicRecommendation'. If you use the code in your research, please cite the paper as:

> Karim M. Ibrahim, Elena V. Epure, Geoffroy Peeters, and Gaël Richard. 2021. Audio Autotagging as Proxy for Contextual Music Recommendation. [*Under Revision*]

## Instructions

We recommend using [Docker](https://www.docker.com/) for reproducing the results of the experiments. 

Clone the repository then follow these instructions.

1. To build the docker image run:
```
docker build -t sit-gen-image docker
```
This will automatically clone the repository inside the docker container

2. Run the container with: 
```
nvidia-docker run -ti --rm --memory=20g --name=sit-gen-container -p 8888:8888 sit-gen-image
```
Note: you might need to adjust the memory or port according to your machine. 

3. Download the dataset available at [Zenodo](https://zenodo.org/record/5552288). 

4. Download the audio previews inside the docker using the [Deezer API](https://developers.deezer.com/api). Then compute the melspectrograms (recommended using librosa) in the direcory "/src_code/repo/spectrograms/" with the following parameters: 
```
"n_fft": 1024,
"hop_length": 1024,
"n_mels": 96
```

5. From the downloaded dataset, create the splits using the 'fold' variable and put them accordingly in the 'groundtruth'. Note: the subdirectories are empty and are meant to guide on how the directories structure should be. 


6. To reproduce our experiments, either run the python file: 
```
python run_all.py
```
The model evaluation results exist in the output directory in 'src_code/repo/experiment_results/'


Or go through the code block by block in the accompanied notebook 'Notebook.ipynb'. To run jupyter use:
```
jupyter notebook --allow-root --ip=$(awk 'END{print $1}' /etc/hosts) --no-browser --NotebookApp.token= &
```
Then access the notebook through the designated port and run all code blocks in order of the following notebook

## Items 
This repository contains the following item: 
- '**run_all.py**' the script for rerunning all experiments in the paper and producing the complete results.
- '**Notebook.ipynb**' Same as the script, but allows for running the code block by block and access the intermediate results.
- '**requirements.txt**' contains the required packages to run the code. Only needed in case docker is not used.   


The repository contains two directories: 
- '**Docker**' contains the script for building the docker image and installing the requirements.
- '**groundtruth** contains the subdirectories in the same format needed to rerun the code. Note: Dataset need to be downloaded seperatly before splitting and moving to the corresponding subdirectory. 


## Acknowledgment
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068.

