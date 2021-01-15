# Detect-and-count-differnet-products using faster rcnn

This end to end project is to detect the products and perform count on different products scanned.

Problem statement: In grocery stores, sometime customers pick products and keep them on different place(not on designated space on the shelf) due to some reasons like he or she changed the mind or price was high. Store user need to find these products and place them on right place on the self, user may need to scan all the products one by one to find the correct location on the self.

Solution: instead of scanning bar code of each products one by one, i have designed an object detection model which will display the products with correct level and also give count of each products.
Model has been trained using Faster RCNN on 32 different products and approx. 500 images have been used for training.
Deployment is done using flask
For retraining the model, need to add the new images in the input file and perform steps like python xml_to_csv.py, generate tfrecords for both train and test data, however no need to train from starting, we can use model weight using the saved checkpoint and model will start from the next step saved in checkpoint.

Download train and test data from google drive https://drive.google.com/drive/folders/1S_K46byzFWi8KfZoxHoFnFi1cKjNUUZc. images are already levelled using labelling tool labelImg, if need to add new data, run this tool for them.

Steps for training:
#Create environment 
conda create -n product_shelf python=3.6   
### Install all requirements 
pip install -r requirements.txt 

#Go inside training folder and run below commands
protoc object_detection/protos/*.proto --python_out=.

python setup.py install

under object_detection folder run object_detection_tutorial.ipynb file to test if everything is fine.

#Run below to convert XML to .CSV file, this is inside training folder
python xml_to_csv.py

Edit generate_tfrecord.py file with all the different levels used for detection
#Run below to generate tfrecords for train and test data.
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

go to training folder
 : change faster_rcnn_v2_coco.config file with correct levels
 : edit labelmap.pbtxt file with all labels

come back :
#Train model with below command
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
 if fail update protobuf  pip install --upgrade protobuf
## To use model first need to stop training and generate inference graph using below py file
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-646 --output_directory my_model
#here 646 is last saved checkpoint
## To restart the training run below py file to take backup and restart the training, it will start after the last saved checkpoint 
python copy_dir.py my_model my_model_backup
## Deploy the model
I have deployed the model on local device only using laptop webcam.
Run product_detection.py file 
It will display the products with label and also display how many of each products in the frame. 




