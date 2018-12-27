# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/distribution.png "Visualization"
[image2]: ./examples/gray.png "Grayscaling"
[image3]: ./test/random.png "Random exploration"
[image4]: ./test/animal_crossing.jpg "Traffic Sign 7"
[image5]: ./test/end.jpg "Traffic Sign 8"
[image6]: ./test/general%20caution.jpg "Traffic Sign 9"
[image7]: ./test/priority.jpg "Traffic Sign 10"
[image8]: ./test/speed%20limit%2030.jpg "Traffic Sign 11"
[image9]: ./test/speed%20limit%2050.jpg "Traffic Sign 7"
[image10]: ./test/speed%20limit%2060.jpg "Traffic Sign 8"
[image11]: ./test/stop.jpg "Traffic Sign 9"
[image12]: ./test/yield.jpg "Traffic Sign 10"
[image13]: ./test/prediction.png "Prediction"
[image14]: ./test/prob.png "Probability"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yz540/traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Each label of a traffic sign class is shown here.
![Labels][image3]

The distribution of each class is shown in the graph below, a bar chart showing the number of data for each class. The x-axis denotes label and the y-axis shows the number of images for each label.

![Data distribution][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because grayscale keeps most of the original image information, but has only one channel and reduces computational requirements compared to 3-channel RGB images. Here is an example of a traffic sign image before and after grayscaling.

![grayscale][image2]

As a last step, I normalized the image data using (pixel - 128)/128 because it will change feature values to be within [-1, 1] which makes the optimization of loss easier.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers as in the 7th cell:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout				| keep probability 0.5        									|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 24x24x16       									|
| RELU					|												|
| Dropout				| keep probability 0.5        									|
| Max pooling	      	| 2x2 stride,  outputs 12x12x16 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x30 	|
| RELU					|												|
| Dropout				| keep probability 0.5        									|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 8x8x48       									|
| RELU					|												|
| Dropout				| keep probability 0.5        									|
| Max pooling	      	| 2x2 stride,  outputs 4x4x48 				|
| Fully connected		| input 768, output 200     									|
| RELU					|												|
| Dropout				| keep probability 0.5        									|
| Fully connected		| input 200, output 43     									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer that implements the Adam algorithm with batch size 120, 10 epochs and a global learning rate 0.002.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.980
* validation set accuracy of 0.951
* test set accuracy of 0.939

At first, I used the well-know LeNet-5 used in the course. It has shown good performance in the MINST example used in the course. But the validation accuracy of this project didn't reach 90%. After normalizing and grayscaling images, the validation accuracy reached 94%. But it misclassified one of the web images, the animal crossing sign, as slippery road sign. 

In the LeNet-5, each convolution layer is followed by a pooling layer, which reduces the information in the data. I added more convolution layers and dropout layer to have high accuracy and reduce over fitting. I tuned the batch size, convolution kernel size and learning rate.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 9 German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12]

The original images contain noises, and the model didn't manage to predict them correctly. Then I cropped them to keep only the traffic sign parts. They are then resized to 32x32x3 for the input using the code:
```cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA )
```
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![Prediction][image13]

The model was able to correctly guess 9 of the 9 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.9%. But one critical drawback of this model is that it cannot recoganize the intrest part in an image by itself.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.
This image shows the top five soft max probabilities for these images.
![Probability][image14]


