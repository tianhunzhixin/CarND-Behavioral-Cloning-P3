#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Nvidia-architecture.png "Model Visualization"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on [NVIDIA architecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with 3x3 filter sizes and depths between 16 and 64 (model.py lines 89-101) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 84). At the input of the model, I use a Cropping layer to crop height and width of the image by 90 px and 20 px respectively.

####2. Attempts to reduce overfitting in the model

The model contains max pooling layers in order to reduce overfitting (model.py lines 90-94). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 40-74). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the center track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 103).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to choose a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate. The model construct is below:

First I start a Sequential model and directly add a lambda layer to normalize the data for input shape of image size that is 160x320x3.
Then I use a cropping layer to cut 75 pixels from top that is basically sky, tree, etc and to cut 25 pixels from bottom that is filled with car itself, and cut 20 pixels from left and right side each that is no use for train. The advantage is that increases training speed by decreasing the data size and clear the data we don't care about.

After these layers, the model has 3 convolutional layers with filter size of 3X3 that increases output depth to 16, 32 and 64 in each layer. All of these convolutional layers have Relu activation and also followed by maxpooling with size of 2x2. The maxpooling layers help keep the system away from overfitting. Then I have a flatten layer followed by 4 fully connected layers that decreases the depth from 400 to 100, from 100 to 20 and from 20 to 1 that is final output.

Final output is the angel that we need for driving the car. And I use Adam optimizer to minimize Mean Squared Error as our criteria for minimizing the error in output.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track from no line place. To improve the driving behavior in these cases, I collect the different case of driving in the place that is no driving line.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

* Lambda layer for normalization. Input 160x320x3. Divides the values by 255 and adjust the average by 0.5.
* Cropping layer: Input 160x320x3, cropping 90 px height, and 20 px width.
* Convolution layer: 16 filters with 3x3 kernel. Activation: RELU.
* Maxpooling layer: size is 2x2
* Convolution layer: 32 filters with 3x3 kernel. Activation: RELU.
* Maxpooling layer: size is 2x2
* Convolution layer: 64 filters with 3x3 kernel. Activation: RELU.
* Maxpooling layer: size is 2x2
* Flatten layer.
* Fully connected. Output: 1400. Activation: RELU.
* Fully connected. Output: 100. Activation: RELU.
* Fully connected. Output: 20. Activation: RELU.
* Fully connected. Output: 1.


Here is a visualization of the architecture:

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from error.

Then I repeated this process on track two in order to get more data points.

Next I recorded the vehicle recovering from the no driving line place to center so that the vehicle would learn to recover from error.

To augment the data sat, I also flipped images and angles. And I use the three camera data for train model.

The solution of data can be found in section 1 "Solution Design Approach".
