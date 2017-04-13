# self_driving_car
## Vehicle Detection and Tracking
- Built a software pipeline to identify vehicles in a video from a front-facing camera on a car.
- Trained and tested a LinearSVM classifier to differentiate a car image from a non car image. Applied the classifier across each frame in the video, sampling small patches along the way to detect cars.
- The classifier contained 8460 HOG and color features in YCrCb color space, extracted from 8792 car images and 8968 non-car images in GTI data set. The accuracy of the classifier was 98.99%.

## Behavior Cloning
- Trained and evaluated a deep learning model to drive a car safely and smoothly on the road in the simulator, with no tire leaves the drivable portion of the track surface.
- Both center lane driving, and recovery driving behavior were recorded to ensure the deep learning model observe behavior in different situations.
- The 3-layer convolutional neural network model was implemented with Keras on a GPU cluster in Amazon Web Services.
