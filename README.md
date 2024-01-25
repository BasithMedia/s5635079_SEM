
# Disease Classification
Basith Shuhaib (ID: 5635079)

#### To determine the specific disease category indicated in photos of paddy leaves.

![depositphotos_123427332-stock-photo-green-rice-field](https://github.com/BasithMedia/s5635079_SEM/assets/149077310/14072839-aeb3-4b8a-8ea0-a171e945c494)

Figure 1: stock-photo-green-rice-field (TZIDO 2015)

### Abstract
Rice, scientifically known as Oryza sativa, is a globally consumed staple food. Paddy, the unprocessed grain prior to husk removal, is cultivated primarily in tropical regions, particularly in Asian nations. Continuous monitoring is essential for rice agriculture due to the susceptibility of paddy crops to various diseases and pests, which can result in a significant reduction of up to 70% in crop production. Professional oversight is typically required to alleviate these infections and avert agricultural damage. Due to the scarcity of crop protection specialists, the process of manually diagnosing diseases is laborious and costly. Therefore, it is becoming more crucial to automate the process of identifying diseases by utilising computer vision-based techniques that have demonstrated promising outcomes in many fields.[2]

## Introduction
The primary goal of this challenge is to create a machine or deep learning model that can accurately classify the provided paddy leaf photos. The training dataset consists of 10,407 labelled photos, which account for 75% of the total. These images are divided into ten classes, including nine illness categories and one class for normal leaves. Furthermore,  data for every photograph, including details about the type of paddy and its age. objective is to categorise each paddy image in the included test dataset, which consists of 3,469 (25%) photos, into one of the nine disease groups or as a normal leaf.[2]

![Dataset](https://github.com/BasithMedia/s5635079_SEM/assets/149077310/902568ab-0e4b-4a1f-881c-9bcdea864a85)

## Literature Review

According to the research (Petchiammal et al. 2022), Paddy farmers encounter significant biotic stress factors, including infections caused by bacteria, fungi, and other organisms. These diseases have a serious impact on the health of plants and result in substantial crop loss.
The majority of these illnesses can be detected through regular observation of the leaves and stems under the guidance of an expert. Identifying paddy diseases manually is a difficult task in a country with extensive agricultural areas and a shortage of crop security specialists. In order to address this issue, it is imperative to automate the process of identifying diseases and offer readily available decision support tools to facilitate efficient crop protection measures. Nevertheless, the limited accessibility of public datasets including comprehensive disease data hinders the feasible deployment of precise disease detection systems.[4]


## Dataset Description and Visualisation
The training dataset consists of 10,407 labelled photos, which account for 75% of the total. These images are divided into ten classes, including nine disease categories and one category for normal leaves. In addition, the system included supplementary metadata for each photograph, including the type of rice and its age. 


There are nine diseases and one normal.

- Bacterial leaf blight
- Bacterial leaf streak
- Bacterial panicle blight 
- Blast 
- Brown spot
- Dead heart
- Downy mildew
- Hispa
- Tungro 
- Normal

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Model Description

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

A basic convolutional neural network (CNN) was used as the model. The model is described in in depth as follows:

1. Input Layer:
- The model receives a 3-channel picture (RGB) with varying dimensions for both height and width as input.

2. Convolutional Layers:
- The model initiates with three convolutional layers, namely conv1, conv2, and conv3, using Rectified Linear Unit (ReLU) activation functions.
- The conv1 layer takes input images and applies 64 filters with a kernel size of 3x3, producing 64 feature maps.
- The conv2 function takes the output from conv1 and adds 128 filters with a kernel size of 3x3, resulting in 128 feature maps.
- The conv3 layer receives the output from conv2 and applies 256 filters with a kernel size of 3x3, resulting in 256 feature maps.

3. Pooling Layers:
- After each convolutional layer, the feature maps undergo max-pooling using pool1, pool2, and pool3 layers. The max-pooling operation reduces the spatial dimensions of the feature maps by half, using a 2x2 kernel size and a stride of 2.

4. Fully Connected Layers:
- Following the convolutional and pooling layers, the feature maps undergo flattening, resulting in a 1D tensor.
- A fully connected layer, denoted as fc1, is utilised with 512 neurons and a Rectified Linear Unit (ReLU) activation function.
- Lastly, a fully connected layer called fc2 is used to transform the output of fc1 into the desired number of output classes.

5. Output Layer:
- The output layer consists of num_classes neurons, where num_classes is the number of unique labels in the dataset.
- The model uses a softmax activation function to generate class probabilities for every input image.

6. Training:
- The model is trained using the cross-entropy loss function and the Adam optimizer with a learning rate of 0.001.
- A learning rate scheduler (ExponentialLR) is used to adjust the learning rate during training.

Overall, this CNN architecture is relatively simple, consisting of multiple convolutional layers followed by max-pooling layers for feature extraction, followed by fully connected layers for classification.


## Results and Discussion

Epoch 1/25
Learning Rate: 0.0010000

Train Loss: 2.6537 Acc: 0.2390
Val Loss: 1.9340 Acc: 0.3612
.
.
.
.

Epoch 25/25
Learning Rate: 0.0002920

Train Loss: 0.9160 Acc: 0.7034
Val Loss: 0.6764 Acc: 0.7848
Training complete in 18m 57s
Best Validation Accuracy: 0.7877

Dataset and Data Loading:

The dataset has photos classified into distinct categories reflecting different diseases.
The loading of images is accomplished by use the OpenCV and PIL libraries, while a custom dataset class called PaddyDataset is developed to manage the loading and preparation of data.

Training:

The model is trained using the training dataset, employing data augmentation techniques such as random resizing/cropping and horizontal flipping. The training process consists of 25 iterations, each using a batch size of 64. Learning rate scheduling and early stopping are utilised to enhance training efficiency and prevent overfitting.

![TRAIN LOSS](https://github.com/BasithMedia/s5635079_SEM/assets/149077310/ee40402c-0236-4089-8774-5a220b2b75db)


![TRAIN ACC](https://github.com/BasithMedia/s5635079_SEM/assets/149077310/88f3559e-a9e6-4da8-b13b-dde3a04ae0dc)


## Conclusion

In conclusion, the applied CNN model demonstrates encouraging outcomes in categorising disease classifications in photos of paddy plants. To enhance performance,  investigate alternative model topologies, hyperparameters, and perhaps integrate advanced methods such as transfer learning. Furthermore, the act of visualising misclassified samples and examining model predictions can offer valuable insights into potential areas that require further improvement.


## References

[1].	Arjunan, P., 2022. Paddy Doctor dataset [online]. Paddy Doctor. Available from: https://paddydoc.github.io/dataset/ [Accessed 22 Jan 2024].

[2].	Arjunan, P. and Petchiammal, 2022. Paddy Doctor: Paddy Disease Classification [online]. kaggle.com. Available from: https://kaggle.com/competitions/paddy-disease-classification [Accessed 21 Dec 2023].

[3].	konradszafer, 2022. Paddy Disease, PyTorch (acc=98.0%) [online]. Kaggle.com. Available from: https://www.kaggle.com/code/konradszafer/paddy-disease-pytorch-acc-98-0/notebook [Accessed 22 Jan 2024].

[4]. Petchiammal, Briskline Kiruba, Murugan and Pandarasamy Arjunan, 2022. Paddy doctor: A visual image dataset for automated paddy disease classification and benchmarking [online]. Proceedings of the 6th Joint International Conference on Data Science & Management of Data (10th ACM IKDD CODS and 28th COMAD). Available from: https://api.semanticscholar.org/CorpusID:254018440.

[5]. TZIDO, 2015. Paddy rice field, nature background [online]. Depositphotos. Available from: https://depositphotos.com/photo/paddy-rice-field-nature-background-32810187.html [Accessed 22 Jan 2024].



