# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/dataset_hist.png "Dataset histogram by classes"
[image2]: ./examples/classes.png "Classes examples"

[image3]: ./examples/3.png "60 km/h"
[image4]: ./examples/7.png "100 km/h"
[image5]: ./examples/12.png "Priority road"
[image6]: ./examples/13.png "Yield"
[image7]: ./examples/41.png "End of no passing"

[image8]: ./examples/activations_1.png "60 km/h Conv1 activations"
[image9]: ./examples/activations_2.png "100 km/h Conv1 activations"
[image10]: ./examples/activations_3.png "Priority road Conv1 activations"
[image11]: ./examples/activations_4.png "Yield Conv1 activations"
[image12]: ./examples/activations_5.png "End of no passing Conv1 activations"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I the shape property of numpy arrays and numpy's unique method to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To perform preprocessing of the images I convert them to grayscale since it is shown to be benefitial when dealing with this challenge, and I standarize them. The standarization step fits the dataset to a normal distribution of mean 0 and standard deviation of 1.

To augment the training set, I perform random rotations and horizontal flipping (this can be seen in the data_generator function under the train section of the notebook). This augmentation transforms the dataset from 34799 samples to 5011056 samples with horizontal flipping and random rotations of 0 to 180 degrees.

This step helps the network generalize much better, making it more robust. Additional augmentations could be horizontal or vertical shifting.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model looks like a VGG16 network. It is a simple feedforward model with dropout regularization.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	     	| 2x2 stride,  outputs 16x16x32 				|
| Dropout		     	|  												|
| 				     	|  												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	     	| 2x2 stride,  outputs 8x8x64	 				|
| Dropout		     	|  												|
| 				     	|  												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Max pooling	     	| 2x2 stride,  outputs 4x4x128	 				|
| Dropout		     	|  												|
| 				     	|  												|
| Fully connected		| 2048 units   									|
| Fully connected		| 1024 units   									|
| Fully connected		| 128 units   									|
| Fully connected		| 43 units   									|
| Softmax				| 	        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The loss function I use in this case is Softmax Cross Entropy.

It is worth mentioning that the softmax layer in the model is applied after the optimizer. The reason for doing this is because the softmax_cross_entropy_with_logits the logits uses directly to calculate the loss. Therefore, we only apply softmax manually when we need the probabilities.

I picked the Adam Optimizer for its properties of momentum and adaptability of the learning rate.

I also defined a data_generator to feed the images to the model when training since it would be impossible to load the entire augmented dataset into memory.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.60 %
* validation set accuracy of 96.21 %
* test set accuracy of 94.25 %

The first architecture I tried was based on the paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). I connected the output of the first and second phases to the fully connected layer together with the output of the third one. I chose this architecture as it is believed to improve the results by giving spatial information to the classifier. 

However, I found that the network was difficult to train. Maybe I implemented it the wrong way. Therefore, I moved onto simple feed forward models. The first network underfitted the data because it was very simple, and when I increased the depth of the convolutional layers, the model had a hard time training, since the dataset is not very big.

I ended up trying to imitate the VGG16 architecture, with 3 phases of 2 convolutional layers with relu activations, followed by max pooling. I thought this architecture could work in this case because it has not lots of layers like ResNet152, or ResNet50. As a result, I thought it should be trainable with the German Traffic Sign Dataset.

It is worth metioning that to avoid overfitting the data due to the complexity of the model, I added dropout. Both in the convolutional network and the classifier. Specifically with 50% and 30% probability of dropping the activations respectively.

This architecture layed down a 96% accuracy on the validation set. In addition to that, trying to classify images it had not seen, the network did very well since these images where hand-picked to be difficult.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6] 
![alt text][image7]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km/h      			| 60 km/h   									| 
| 100 km/h     			| 100 km/h 										|
| Priority road			| Priority road									|
| Yield					| Yield											|
| End of no passing		| End of no passing    							|


I picked the "End of no passing" sign since it showed to be the worst ROC curve among all the signs. In addition, I picked I blurry image. I also picked dark images and, for instance, the priority road image with some lightning in the background that could mislead the classifier. Nevertheless, the network classified all of them correctly with high confidence.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all the images, the confidence was of +99% probability for the correct class.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99 	     			| 60 km/h   									| 
| .99 	     			| 100 km/h 										|
| .99 					| Priority road									|
| .99 					| Yield											|
| .99 					| End of no passing    							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I would say the network pays attention to the shapes of the signs. For example the priority road and yield signs differ a lot in activations compared to the round signs. It also looks at the numbers in the case of 60 km/h and 100 km/h signs.

#### 60 km/h
![alt text][image8]
---
#### 100 km/h
![alt text][image9]
---
#### Priority road
![alt text][image10]
---
#### Yield
![alt text][image11]
---
#### End of no passing
![alt text][image12]