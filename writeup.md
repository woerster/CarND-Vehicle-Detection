
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./output_images/test_images/test1.jpg
[image4]: ./output_images/test_images/test6.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `extract_features()` (defined in lines 62 and following, called in lines 270 and 276 in file `pipeline.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images (beginning in line 244).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

The the `extract_features()` function is applied on the two image sets. The function takes several parameters to steer what and how features are extracted from the given images. 

#### 2. Explain how you settled on your final choice of HOG parameters.

The parameter values in the code for lesson 34 (see lesson code, file `search_classify.py`, lines 115 following) give a good starting point to test the performance of the classifier with varying parameters.

I tested different color spaces, changed the value of the hog parameters (halving and doubling) and raised the number of spacial feature, one by one and checked how the classifier performed. 

I settled down with the values set in lines 259 to 268 in file `pipeline.py`, when the classifier reported a test accuracy of around 0.98. So, I did not check the influence of the color and spacial features on the overall performance. It could very well be that the HOG features are good enough for the car detection task. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in line 305 in `pipeline.py`. Before, I scaled and normalized the extracted (non-)car features and split a randomized collection into a train and a test set with a 20% test split size.

The reported test accuracy was around 0.98%. This seems to be good enough for the car detection task.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is performed in the function `find_cars()` in lines 137 through 205. The search itself is done in the for loop starting in line 169.

I decided to search in windows in two scales (2.0 and 1.0) in the lower half of the camera frame. Cars in the sky are not so common nowadays ;) . The smaller scale performed better in the detection of cars that are farther away. The bigger one performed better on cars that were nearer the front camera. 

The overlap is defined in terms of cells per step. In the current version I use a step size of 2. With a block size of 8 cells per block this makes up for a 75% overlap of the windows.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
![alt text][image4]

The left images show the camera frame overlayed with the windows that the classifier detected to be a part of a car (blue boxes). Additionally, the cyan boxes are a bounding box of the summed and thresholded windows (for details see the description of the heatmap filter for false positives in the sections below).

As written earlier, I modified (halving and doubling) the initial HOG parameter values and looked at the reported classifier accuracy. 

For the car detection, I started to use only one call of the `find_cars()` function with varying scales to see how the detection performs for the test images. I finally settled at a combination of scales (1.0 and 2.0) which seems to work fine on test images and the test and project videos.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video and stored these in a deque of detections per frame. Then, I created a heatmap by summing all the detections of the last 10 camera frames and thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Examples of the output of the heatmap algorithm can be found on the right images above.

The heatmap algorithm is implemented in file `pipeline.py` beginning on line 342.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The determination process for a good set of parameter values for the linear svm was a pretty manual process. This could be done in a more automatic way where a lot of combinations are tested subsequently without manually editing the values. And then the best performing set of parameters with the highest accuracy is used. 

The threshold of the heatmap algorithm is a somewhat arbitrarily chosen value. It seems to work fine for the given videos.

Problems for the given classifier will be differing weather conditions (fog, rain, etc.) Additionally dirt on the road and the camera optics will prevent a good performance of the car detection.

Also, the very different lighting conditions through night and day will make it difficult for the algorithm to work properly.
