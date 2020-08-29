# Fire Detection using CCTV images — Monk Library Application

## Goal
* Make a fire-detection deep learning model using the Monk library and transfer learning.
* Make a webapp to detect fire in a image.

## About the Dataset.
* As the dataset of CCTV images were very less. Therefore I took 2 datasets from Kaggle.
* The first one contained images of fire and non-fire, and the second one contained CCTV images of fire and non-fire.
* Then I merged these two datasets for greater accuracy.

## Table of contents in Jupyter Notebook.
*  Install Monk
*  Using the pre-trained model for fire detection
*  Training a classifier from scratch
    * Loading the Dataset
    * Importing Keras backend
    * Training model using mobilenet_v2 as transfer learning architecture
    * Training model using desnenet121 as transfer learning architecture
    * Training model using densenet201 as transfer learning architecture
    * Comparing all the models.
*  Conclusion

Visit the [blog](https://medium.com/p/242df1fca2b9) for more details.

![Output](https://github.com/rohit0906/fire-detector/blob/master/static/linked-post.gif)

## Make web application using Flask library.
![Webapp](https://github.com/rohit0906/fire-detector/blob/master/static/webapp.gif)
