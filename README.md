# This is simple image classification ML model which uses traditional techniques
## process
- loading the cifar file manually as the attempt to load cifar file using tensorflow failed
- loading datas/labels to the variablesw x/y
- reshaping and transpose of data and labels,convert to greyscale to reduce the computing
- resizing the images for faster computation
- flatten image for computation as ml model understands numbers(1D numbers) only
- splitong the data for training and testing
- creates models,training the model
- prediction,giving accuracy,confusion matrix
- ploting three images original,brightness adjusted,gaussian blur
- plotting a predicted and respective answer of testing models
# output gained
-Accuracy 28.12% 
-wrong ouput redicted beacause of the low quality of training data, it should be in low quality else my system will not able to compute
# Requirements
-pickle,numpy,cv2,os,sklearn.svm,sklearn.metrics,sklearn.model_selection,matplotlib.pyplot
-need to place cifar-10 python file in the project folder
