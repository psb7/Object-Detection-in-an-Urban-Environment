# Object Detection in an Urban Environment

## Goal

To classify and localize the cars, pedestrians and cyclists for a given image frame.

## Data

The project will be making use of a public dataset: [Waymo Open dataset](https://waymo.com/open/).


## Structure


### Data

The data we will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: with 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video.




## Local Setup

For local setup to train with Nvidia GPU, the Dockerfile and requirements is in the [build directory](./build).

Follow [this README](./build/README.md) to create a docker container and install all prerequisites.


### Download and process the data


**Disclaimer:** As I had used Udacity workspace for this project, dataset was already available. To download the dataset manually please follow the steps defined in the next paragraph.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, only a subset of the data (images) is required. Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

The script is run using following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

Downloading data may take some time.



## Dataset


### Exploring dataset


| ![](assets/eda0.png)  |  ![](assets/eda1.png) |
:-------------------------:|:-------------------------:
| ![](assets/eda3.png)  |  ![](assets/eda4.png) |



### Analysis


Random five thousand samples from the dataset are used for analysis.

 Dataset is very skewed in terms of number of samples available for each class. Class 1 of cars have maximum samples. Class 4 of cyclists has very less occurrences in the dataset. Whereas,class 2 of pedestrians falls in between.

 ![Classes vs counts](assets/eda2.png)





### Create the training - validation splits

Split the data into 87% train, 10% validation and 3% test.
Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### The config file


The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.



### Training

Model and training hyperparameters are defined using a file, pipeline_new.config.
You can make changes in this config file, then move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.








### Augmentation

Explored the Object Detection API and applied many different augmentations

Used various augmentation strategies:
1. random_horizontal_flip
2. random_crop_image
3. random_adjust_brightness
4. random_distort_color
5. random_adjust_saturation
6. random_black_patches

The reulting images from the Augementations are as shown below:

| ![](assets/aug1.png)  |  ![](assets/aug2.png) |
:-------------------------:|:-------------------------:
| ![](assets/aug3.png)  |  ![](assets/aug4.png) |




### Experiments
1. Initially the training performed on with the configuration file with default setting. The results are as shown in the figure below.
![](assets/def_metrics.jpg)
![](assets/map_def.jpg)
![](assets/ar_def.jpg)
![](assets/lr_def.jpg)

  As seen from the figure there no much improvements from the training over time.

2. For the second experiment the hyperparameters: batch size, learning rate, optimizer, and augmentations as defined above are modified.With batch size = 4, optimizer = adam and learning rates: 0.0005 for first 400 steps, 0.0001 for steps  400 - 999, 0.00008 for steps 1000 to 1499, and there after 0.00004. The modification is as shown in the figure below.

  ![Augementations](assets/mod1.jpg)
  ![optimizer](assets/mod2.jpg)

  ![](assets/mod_metrics.jpg)
  ![](assets/map_mod.jpg)
  ![](assets/ar_mod.jpg)
  ![](assets/lr_mod.jpg)



### Conclusion
Clearly the experiments reveals that the modification performed as defined above has really helped the object detection model to perform well. The comparison between errors and metrics for both experiments also states the same.

Hence it is possible to conclude that selection of batch size, optimizer, augmentations, and learning rates definitely has improved the model performance.


### Creating an animation

#### Export the trained model

Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path data/waymo/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Visualisation

![](assets/animation.gif)
