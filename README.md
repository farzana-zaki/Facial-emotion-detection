# Facial emotion detection using ANN, CNN and Transfer Learning

Facial emotions and their analysis are essential for detecting and understanding human behavior, personality, mental state, etc. Most people can recognize facial emotions quickly regardless of gender, race, and 
nationality. However, extracting features from face-to-face is a challenging, complicated, and sensitive task for the computer vision techniques, such as deep learning, image processing, and machine learning to
perform automated facial emotion recognition and classification. Some key challenges in extracting features from the facial image dataset include a variation of head pose, resolution and size of the image, background,
and presence of other objects (hand, ornaments, eyeglasses, etc).In recent years, deep learning has become an efficient approach with the implementation of various architectures which allow the automatic extraction of
features and classification using convolutional neural network (CNN), transfer learning, and recurrent neural network (RNN). This project aims to build a CNN model for facial emotion detection accurately.

## Objective: 

The purpose of this project is to build a CNN model for facial emotion detection accurately. The proposed CNN model performs multi-class classification on images of facial emotions to classify the expressions
according to the associated emotion.

## Solution summary:

The proposed CNN model performs multi-class classification on images of facial emotions to classify the expressions according to the associated emotion.

In this facial emotion detection project, various CNN models (simple, transfer learning and complex) are employed for training, validation and testing to observe the accuracies to detect those emotions.
In total, 17 different configuration CNN models are applied and evaluated. Simple and transfer learning CNN models are overfitting and has low F1 score.

For building the proposed CNN model, the hyperparameter tuning using random search from the keras tuner was applied to select the building blocks of the complex CNN models. As an optimizer, adam with three various
learning rates: 0.1,0.01 and 0.001 are used. A layer with five convolutional blocks for feature selection and three dense layers for the classification are used for the complex CNN models with batch size of 16,32
and 64. Out of three complex CNN models, model 6a(CNN model with five convolutional blocks for feature selection and three dense layers for the classification, with batch size of 32, learning rate of 0.001 and adam
optimizer) shows the best performance. Model 6a is selected as the final proposed CNN model for the face emotion detection.

The final proposed model solved the overfitting problem and is well generalized and optimized with training, validation and overall test accuracies of 72.23%,69.10% and 74%, respectively. This model has achieved an
average F1 score of 0.74. Batch normalization and drop out are used to solve the overfitting problem.

However, the model has poor performance for detecting class-1 (neutral) and class-2 (sad) with F1 score of 0.70 and 0.56, respectively.

## Key recommendations and future implementation:

(1)Training dataset is slightly imbalanced. However, validation dataset is pretty much imbalanced.Total number of images of the four classes (Happy:1825, Neutral:1216, Sad:1139, Surprises:797) in the validation dataset.
As we can see that 'surprise' and data has less frequency (0.16) compared to other three emotions. So, validation dataset is imbalanced due to the 'surprise' dataset. We could employ oversampling technique for the
'surprise' dataset to make the balanced dataset and then again train the model and compare the performances.

(2)Also, there are some poor quality images in the training dataset. For example, some images contain watermarked text, some training images does not have any facial expressions (rather has question marks or cross
sign instead of any image).Images of neutral and sad faces are pretty much confusing. Therefore, CNN algorithms has faced the difficulty to correctly detect them properly. Therefore, for all the classifiers, F1
scores of neutral and sad emotions are not satisfactory.

(3)The dataset is pretty small. Data augmentation can be applied to generate a large volume of training dataset by using the transformations of the face images, such as flip, shift, scaling, and rotation.

(4)Several experiments needs to be carried out with mode convolutional layers (such as 6 or more layers) to verify the effectiveness of the augmented dataset, and the performance of these approached CNN models in
comparison with some of the frequently used face recognition methods.

(5)The proposed CNN model was implemented using GPU. Using more deeper convolutional layers with millions to trillions of training dataset may increase the implementation cost.

(6)Several other transfer learning models can be applied to improve the performance of the facial recognition.

(7)It is recommended that stakeholders consider these variables in building improved long-term facial emotion detection models, as well as include the full range of environmental implications of various data
sources, usage of color images instead of grayscale images, usage of more training data in developing future facial emotion detection.


