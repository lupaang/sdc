### Project overview

### Dataset

#### Dataset analysis

The image set contains diverse set of images of which some are blurry, clear, light and some are dark. The function of randomly displaying 10 images, as shown below, is implemented to check whether the image associated its corresponding bounding boxes and class labels are presented correctly.

![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/display_imgs.png)

The bounding boxes are red for the vehicles, green for the cyclists and blue for the pedestrians.

[![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/im_4.png)](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/im_4.png)[![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/im_6.png)](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/im_6.png)

[![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/im_8.png)](https://github.com/abhilash1910/nd013-c1-vision-starter-main/blob/master/images/img1.png)[![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/im_1.png)](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/im_1.png)

We have taken 20000 images from dataset for analyzing: 
- Number of objects for each classs.
- Number of bounding boxes.
- Average brightness.
- Number of images taken in different scenes (day/night).
-----------
> In the image shown below, we can see the number of  `car`  are much more than  `pedestrian`  and  `cyclist`. The main reason might be the fact that most of images are taken from car driving not in downtown, Thus there are not many people actually appear in the scene. The number of  `pedestrian`  and  `cyclist`  for training a model may not be sufficient.

![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/class_distribution.png)
-----------
> Now we look at the statistics of driving scenes for daytime/night. We calculate each  pixels' average value over 3 channels(RGB) `np.sum(np.sum(img))/(640*640*3)`. If the average value is less than 50, we identify the image is taken at night. The results showed that the number of images taken in daytime is dominant, which make it more challenging for recognizing objects in the darker scenes.

![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/scences.png)
----------
> Next, we analyze the bounding box's total count and the distribution of the objects (bounding boxes) in images. As shown below, we can see that most of images are in 20 objects per image. It may imply that some of the objects are overlapped, which means only partial information is captured by the camera.
![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/object_per_image.png)

#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training

### Experiment 1 (Reference experiment)

We perform the transfer learning using [SSD_ResNet50 model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) with the [default pipeline configuration](https://github.com/lupaang/sdc/nd013-c1-vision-starter/experiments/experiment_1/pipeline_new.config). The results of Loss and DetectionBox_Recall/Precision will be served as baselines. The curve in orange is Loss in training steps and blue dot is Loss in evaluation. The classification loss between training (0.1482) and evaluation (0.3724) is around 0.2242. This indicates that the trained model needs to be more generous to predict objects in unseen data. To improve the initial results, we can add more variabilities in our data to simulate different environments during training. Hence, we will add more options of data augmentation in the pipeline configuration.
![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/experiment_1/loss.png)

**Detection Box precison**: 

![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/experiment_1/detectionBox_precision.png)

**Detection Box Recall**: 

![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/experiment_1/detectionBox_recall.png)

#### Improve on the reference

### Data Augmentation

To improve on the model performance, we try several data augmentation steps such as gray-scale image conversion, random change contrast, saturation, brightness adjustments based on  [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto). Finally, The pipeline changes can be found in [`pipeline_new.config`](https://github.com/lupaang/sdc/blob/main/nd013-c1-vision/experiments/experiment_2/pipeline_new.config) and the experiments with strategies of augmenting data are described in the [next section](#experiment-2). Augmentations applied:
-   0.02 probability of grayscale conversion
-   Brightness adjusted to 0.2
-   Contrast values set to `max_delta`  1.2
-  Saturation values between 0.8 and 1.25
    

|              ||
:-------------------------:|:-------------------------:
![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/augmentated_imgs/augmented_img_1.png)  |  ![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/augmentated_imgs/augmented_img_2.png)
![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/augmentated_imgs/augmented_img_3.png)  |  ![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/augmentated_imgs/augmented_img_4.png)
![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/augmentated_imgs/augmented_img_5.png)  |  ![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/augmentated_imgs/augmented_img_6.png)
----
### Experiment 2

The data augmentation operations done on images are added in data augmentation part in  [pipeline_config](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/experiments/experiment_2/pipeline_new.config):

As shown in the figures below, the difference (0.16) of Classification Loss between training (0.14) and evaluation (0.30) is also better than the baseline. The evaluation metrics, Precision and Recall, also got improved.
**Loss**: 
![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/experiment_2/loss.png)

**Detection Box precison**: 

![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/experiment_2/detectionBox_precision.png)

**Detection Box Recall**: 

![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/experiment_2/detectionBox_recall.png)


## Discussion

There are still a huge room for model preformance. improvement, for example add more augmentated data, adjust more hyperparameter in `pipeline_config`. However, as shown above in data analysis, the class labels (pedestrian and cyclist) are very rare in the dataset. Thus, This is a critical requirement to add more images that contain those labels to have more balanced data and avoid biases which in return will improve accuracy.


## Results

After the trained model is exported, we perform object detection with the model on driving scenes stored in the test set.  
The object detection results are shown in the video below:

[![](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/animation.gif)](https://github.com/lupaang/sdc/blob/master/nd013-c1-vision/images/animation.gif)

The loss is lower than the previous loss (un-augmented model). This is an indication of better performance. We have reduced overfitting to an extent with augmentation, however better classification results would be resulting from a more balanced dataset.