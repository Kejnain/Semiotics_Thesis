# MSc Thesis Project

## Main Directory
- `.gitattributes`: Contains Git LFS (Large File Storage) attributes for pulling large files from the repository. (Refer to `requirements.txt` for more details.)
- `demo.mp4`: Video demonstration showcasing the project's functionality, including its main features and outputs.
- `essay.pdf`: Thesis essay explaining the project's theoretical framework, methodology, and results, along with reflections and conclusions.
- `journal.md`: Weekly journal documenting progress and updates.
- `main.py`: Main pipeline script for running the program.
- `requirements.txt`: Lists requirements for setting up Git LFS, the Conda environment, and necessary library installations.

## Data
- `goemotions.csv`: Dataset from the GoEmotions repository. https://paperswithcode.com/dataset/goemotions
- `scraped_shapes.csv`: Dataset containing images scraped from various sources. https://mrmrsenglish.com/different-shape-names
- `shapes_with_colours.csv`: Dataset with RGB color values added to shapes.
- `shapes.csv`: Final dataset prepared for model training.
- `survey_cleaned.csv`: Processed survey responses, ready for training.
- `survey_responses.csv`: Raw survey responses. https://forms.gle/W7vfHjp9CdYtiNQ36
- `images/`: Directory containing images extracted from URLs in `shapes.csv`.

## Jupyter_Notebooks
- `data_collection.ipynb`: Notebook for data scraping and collection.
- `data_visualisation.ipynb`: Notebook for visualizing and exploring the datasets.
- `pre_processing_survey.ipynb`: Notebook for cleaning and processing survey responses.

## Models
- `sentiment_model.pkl`: Pre-trained sentiment analysis model.
- `vae_model.pth`: Variational Autoencoder model.

## Test 
- `test_sentiment.ipynb`: Test cases and evaluation for the sentiment analysis model.
- `test_vae.ipynb`: Test cases and evaluation for the Variational Autoencoder.


## Train
- `train_sentiment.ipynb`: Notebook for training the sentiment analysis model.
- `train_vae.ipynb`: Notebook for training the Variational Autoencoder.

## Util
- `dataset.py`: Helper functions for loading and managing datasets.
- `gui.py`: Code for graphical user interface components.
- `vae.py`: Implementation of the Variational Autoencoder model.


