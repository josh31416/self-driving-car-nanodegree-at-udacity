# **Behavioral Cloning** 

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/circuit_center.jpg "Circuit"
[image2]: ./examples/road_center.jpg "Road"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_circuit.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the [NVIDIA End to End Deep Learning for Self Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

The images are cropped and normalized in the first two steps of the Sequential model. After that, 3 convolutional layers follow with a 5x5 kernel size and 2 two with 3x3 kernel size. Next, 3 fully connected layers are in charge of acting as the controller for the steering command.

#### 2. Attempts to reduce overfitting in the model

I introduced Dropout at the beggining of the fully connected layer for the network to learn redundant features to predict the steering angle.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, and the learning rate was set to 1e-4.

#### 4. Appropriate training data

Three laps around the track both counterclockwise and clockwise was the data to train the network. I used the three cameras with the steering correction factor to account for recovering situations.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As a first step I implemented a simple fully connected layer to check that everything was working. Once I got that, I tried with an architecture similar to LeNet. However, the results weren't satisfactory. After that, I decided to stick to the NVIDIA architecture. It was design for this problem and the fully connected layer acting as a controller made sense.

After that, validating the model both with the loss and trying it on the simulator, I played with regularization and data augmentation until I got the model working.

For example, the model had trouble initially with the brick bridge or the dirt track. I solved this by gathering more data and applying dropout in the fully connected layer.

Once I got the track, I decided to go for the difficult road. In this case the hills and the shadows were the main challenge. I solved this by augmenting the data tweaking the brightness of the images and applying some transformations.

At the end of the process, the vehicle was able to drive autonomously around the track and the difficult road.

#### 2. Final Model Architecture

The final model architecture is similar to the [NVIDIA End to End Deep Learning for Self Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) expect for the Dropout layer added at the beggining of the fully connected layers.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded three laps on track one using center lane driving. Since I was using the three cameras, there was no need to record recovering behavior examples.

![alt text][image1]
![alt text][image2]

After the collection process, I had 80k samples for the circuit track and 45k for the difficult road.

To augment the data I performed brightness change, zoom, channel shift, height shift and a bit of rotation.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the validation loss when it wouldn't go lower. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### Results

**Circuit video link:** [YouTube](https://youtu.be/KJtsQ8WRiPM)

**Road video link:** [YouTube](https://youtu.be/dxbqPvbprVs)