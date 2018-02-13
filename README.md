## Project: Build a Traffic Sign Recognition Program

### Data Set Summary & Exploration

I used numpy to calculate some basic statistics of the trafficsigns data set:


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

### Design and Test of my Model Architecture

The only preparation of my dataset is that i normalized the image data to values between 0 and 1. Doing this raises the probability of convergation.



#### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 14x14x16  	|
| RELU					| 	        									|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 14x14x16  	|
| RELU					|												|
| Max pooling           | 2x2 stride,  outputs 7x7x16 					|   
| Flatten				| input 7x7x16, output 784 						|
| Fully connected 1600	| input 784, output 120							|
| RELU					|												|
| Fully connected 120   | input 120, output 84							|
| RELU					|												|
| Fully connected 43	| input 84, output 43							|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model i used the categorical crossentropy optimizer with a learning rate of 0.001. I trained the model on 5 Epochs with a batch size of 32, because tinkering with the batch size showed that the learning is way faster with 32 than with a bath size of 64.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.8 %
* validation set accuracy of 97.3 %
* test set accuracy of 88.7 %


* The LeNet architecture was chosen, because it is very good a classifying images of numbers. So my guess was it wouldn't do too bad in classifying traffic signs which essentially vary way less than handwritten digits. But after experimenting a bit settled on a moderately altered version, because the results looked slightly better.
* The modified architecture includes one additional convolutional layer. Additionaly i removed the pooling layer after the second convolutional layer and added it back after the third convolutional layer.
* The validation set accuracy shows that it is indeed able to classify traffic signs. The accuracy on the validation set is just a little less than on the training set. But the accuracy on the test set is pretty bad, which in my opinion shows, that a bigger and more diverse dataset is needed.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.


Here are five German traffic signs that I found on the web:

![alt text][https://github.com/dzeip87/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/30.jpg] 
![alt text][https://github.com/dzeip87/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/60.jpg] 
![alt text][https://github.com/dzeip87/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/70.jpg] 
![alt text][https://github.com/dzeip87/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/Yield.jpg] 
![alt text][https://github.com/dzeip87/CarND-Traffic-Sign-Classifier-Project/blob/master/new_images/keep_right.jpg]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield     			| Yield		   									| 
| 30 km/h     			| 30 km/h 										|
| 60 km/h				| Priority road  								|
| 70 km/h	      		| 30 km/h					 					|
| Keep right			| Keep right      								|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is actually a lot less than the 88.7 % on the test set, but the predictions were not that far off at least in the case of the 70 km/h sign. The bad quality of the downsampled images could be a reason for the poor result. Other than that the test with only five images is not statistically relevant, because of the small number of images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.


For the first image, the model is absolutely sure that this is a Yield sign (probability of 1.0), and the image does contain a Yield sign. So the other probabilities were all at 0.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield		   									| 
| 0.00     				| Ahead only 									|
| 0.00					| Turn right ahead								|
| 0.00	      			| Turn left ahead			 					|
| 0.00				    | No passing      								|


For the second image the model is absolutely sure that it is a 30 km/h sign, which is right. The other probabilities were therefore 0 and of no meaning.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| 30 km/h	   									| 
| 0.00     				| End of all speed and passing limits			|
| 0.00					| Keep right									|
| 0.00	      			| 20 km/h					 					|
| 0.00				    | End of speed limit (80km/h)      				|



For the third image the model is relatively sure (probability of 74.6 %) that it shows a Priority road sign, when it actually is a 60 km/h sign. The next best guess is a 30 km/h sign with a probability 15.6 %. The third highest probability with a value of 9.2 % is an "End of all speed and passing limits" sign. So in this case the model is pretty far off the mark with its prediction, which could be the case because of bad image quality and not enough training.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.746        			| Priority road									| 
| 0.156    				| 30 km/h  										|
| 0.092					| End of all speed and passing limits			|
| 0.005	      			| 60 km/h					 					|
| 0.001				    | Keep right      								|



For the fourth image the model is nearly absolutely sure (probability of 97,6 %) that it is a 30 km/h sign, when it actually is a 70 km/h sign. The second best guess is a "General Caution" sign with a probabilty of 2.4 %.  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.976        			| 30 km/h 	   									| 
| 0.024    				| General caution								|
| 0.00					| Priority road									|
| 0.00	      			| Keep right				 					|
| 0.00				    | 20 km/h 	     								|


For the fifth image the model is absolutely sure that it is a "Keep right" sign (probability of 100 %) which is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Keep right  									| 
| 0.00     				| Go straight or right							|
| 0.00					| Turn left ahead								|
| 0.00	      			| End of all speed and passing limits			|
| 0.00				    | End of no passing								|
