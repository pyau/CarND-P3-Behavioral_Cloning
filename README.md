#**Behavioral Cloning** 

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

[image1]: ./error.png "Training and Valuation error"
[image2]: ./screencap_dash.jpg "Center lane driving"
[image3]: ./right.jpg "Right recovery"
[image4]: ./left_2017_04_19_08_05_21_357.jpg "Left image"
[image5]: ./center_2017_04_19_08_05_21_357.jpg "Center image"
[image6]: ./right_2017_04_19_08_05_21_357.jpg "Right image"

Please click on the image below to view the result of my simulation run.
<a href="http://www.youtube.com/watch?feature=player_embedded&v=8Dbd-E2C1H0
" target="_blank"><img src="http://img.youtube.com/vi/8Dbd-E2C1H0/0.jpg" 
alt="Click to watch video" width="240" height="180" border="10" /></a>

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
python3 drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is an implementation of nVidia's self driving car architecture https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/ (model.py lines 99-119).  In my implementation, I start off with a lambda normalization layer, then a cropping layer, then with 3 convolution layers with pooling, then with 2 convolution layer.  The network is then flattened before applying a dropout layer.  Then I introduced 4 dense layers and return the result. 

The model includes RELU layers to introduce nonlinearity (code line 104, 106, 108, 111, 112), and the data is normalized in the model using a Keras lambda layer (code line 101). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 114). But the model still overfits, as observed in the screenshot.
![alt text][image1]

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 50-57 for data sets, line 126 for splitting data set). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and recordings of driving around curves to ensure the simulation can drive around the whole track.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I first implemented LeNet as a baseline, then focus on getting the simulation working with nVidia's self driving car architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. My model performs better on evaluating the training set, but the difference does not seem very bad.

To combat the overfitting, I modified the model so that it includes a dropout layer.  However, the model still seem to overfit as the second epoch yields higher mean squared error loss than the first for the validation data.

I had implemented fit generator in a separate file clone.py in the same repository.  However, I found that since my computer has enough memory, it is actually faster to not use fit generator.

At the end of the process, the vehicle is able to drive autonomously around the track at 30 mph without leaving the road.

Click on the image below to see the result of the video generation from drive.py output, or you can find the same video in video.mp4 in the same repository.
<a href="http://www.youtube.com/watch?feature=player_embedded&v=f5Exs0wZ7_0
" target="_blank"><img src="http://img.youtube.com/vi/f5Exs0wZ7_0/0.jpg" 
alt="Click to watch video" width="240" height="180" border="10" /></a>

####2. Final Model Architecture

The final model architecture (model.py lines 99-119) consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 160x320x3 RGB image                           | 
| Lambda                | Normalize range from -0.5 to 0.5              |
| Cropping              | Crop image to size 90x320                     |
| Convolution 5x5       | 24 depth, 1x1 stride, same padding, RELU      |
| Max pool 2x2 kernel   | 2x2 stride                                    |
| Convolution 5x5       | 36 depth, 1x1 stride, same padding, RELU      |
| Max pool 2x2 kernel   | 2x2 stride                                    |
| Convolution 5x5       | 48 depth, 1x1 stride, same padding, RELU      |
| Max pool 2x2 kernel   | 2x2 stride                                    |
| Convolution 5x5       | 64 depth, 1x1 stride, same padding, RELU      |
| Convolution 5x5       | 64 depth, 1x1 stride, same padding, RELU      |
| Fully connected layer |                                               |
| Dropout layer         | 0.25x dropout                                 |
| Fully connected       | output 100                                    |
| Fully connected       | output 50                                     |
| Fully connected       | output 10                                     |
| Fully connected       | output 1                                      |
|                       |                                               |

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it drives itself too much to the side. This is how a recovery looks like starting from right side :

![alt text][image3]

To augment the data set, I also flipped images thinking that this would correct the behavior of the car biasing to the left, since this track has more left turns.  I also used the images for left and right cameras.  I use the correction factor of -0.2/+0.2.  Below are a set of left, center and right pictures which are taken at the same time.

![alt text][image4]
![alt text][image5]
![alt text][image6]

I added some training data by recording only driving at the curves.  This proved to be useful for helping the car learn how to make right turns.

I also added several extra laps of driving data for more data points.

After the collection process, I had about 110000 number of data points.  The only preprocess I did was to make sure the images are being feed into my neural network as RGB format.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the plot for error above. I used an adam optimizer so that manually training the learning rate wasn't necessary.
