# AttireClassifier
Attire Classifier is a Cultural Clothes Classifier aimed at creating a CNN Model that can classify clothes from different regions of Pakistan.
## Dataset
Following Dataset was assembled and used:
https://www.kaggle.com/datasets/khizertariq/pakistani-cultural-clothes

## Model Architecture:
For classification a CNN Model was used that predicts outputs for four classes.

## Data Cleaning:
The [converter.ipynb](https://github.com/khizer-kt/AttireClassifier/blob/main/converter.ipynb) can be used to convert files to _jpg_ format while the [datacleaningscript.ipynb](https://github.com/khizer-kt/AttireClassifier/blob/main/datacleaningscript.ipynb) can be used to clean the directories while keeping images of only the allowed formats.

## GradCam:
The Code also uses Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight the regions influencing the decisions of the CNN model thus taking a step towards Explainable AI.
