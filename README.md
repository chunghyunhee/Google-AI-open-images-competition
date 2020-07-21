## Google AI open images competition


- object detection에 있어서 컴퓨터가 정확안 이미지 설명을 제시할 수 
있도록 매우 큰 훈련 set을 제공하여 최첨단 성능을 능하가는 
정교한 물체 및 관계탐지 모델에 대한 연구를 자극하기 위한 대회이다. 
- google AI는 공개 이미지 데이터셋을 공개하여 open image는 현재
전례없는 규모로 PASCAL VOC, ImageNet 및 COCO의 전통을 따른다. 
- open images challenges는 open image 데이터셋을 기반으로 한다. 

--------------------------------------------------------------------------------------
(1) faster_rcnn_inception_resnet_v2_baseline.ipynb
- tensorflow hub에서 mululer를 사용하여 모델과 image를 graph에 넣고 실행
- resnet + Faster RCNN 사용 

(2) ResNet + Faster_R_CNN
- 직접 test image에 적용하여 학습, prediction결과 도출 

(3)
## 1. Dataset
- No external dataset.
- I only use FAIR's ImageNet pretrained weights for initialization, as I have described in the Official External Data Thread.
- Class balancing.
- For each class, images are sampled so that probability to have at least one instance of the class is equal across 500 classes. For example, a model encounters very rare 'pressure cooker' images with probability of 1/500. For non-rare classes, the number of the images is limited.

## 2. Models
- The baseline model is Feature Pyramid Network with ResNeXt152 backbone.
- Modulated deformable convolution layers are introduced in the backbone network.
- The model and training pipeline are developed based on the maskrcnn-benchmark repo.

## 3. Training
- Single GPU training.
- The training conditions are optimized for single GPU (V100) training.
- The baseline model has been trained for 3 million iterations and cosine decay is scheduled for the last 1.2 million iterations. Batch size is 1 (!) and loss is accumulated for 4 batches.
- Parent class expansion.
- The models are trained with the ground truth boxes without parent class expansion. Parent boxes are added after inference, which achieves empirically better AP than multi-class training.
- Mini-validation.
- A subset of validation dataset consisting of 5,700 images is used. Validation is performed every 0.2 million iterations using an instance with K80 GPU.

## 4. Ensembling
- Ensembling eight models.
- Eight models with different image sampling seeds and different model conditions (ResNeXt 152 / 101, with and without DCN) are chosen and ensembled (after NMS).
Final NMS.
- NMS is performed again on the ensembled bounding boxes class by class. IoU threshold of NMS has been chosen carefully so that the resulting AP is maximized. Scores of box pairs with higher overlap than the threshold are added together.
Results.
- Model Ensembling improved private LB score from 0.56369 (single model) to 0.60231.
--------------------------------------------------------------------------------------

#  Object detection 
- 이미지의 classification에서 확장되어, 해당 클래스에서 어떤 위치에 어떤 
물체가 있는지를 보고자 하는 것이다. (실 세계에서의 dection)
- 참고논문 :  https://aaai.org/Papers/AAAI/2020GB/AAAI-ChenD.1557.pdf
- dectection을 한다고 했을 때 R-CNN, Fast R-cnn, Mask R-CNN, YOLO,
YOLO v2 등이 있다.
- 최초가 된 분석 방법이 R-CNN이고, selection search를 보완한 것이 Fast-CNN
이다. 여기서 다시 보완한 모델이 Faster R-CNN이다. YOLO의 경우는
Fater R-CNN보다 속도가 빠르나 예측율이 떨어진다.
- 결국 Yolo와 Fater R-CNN은 trade off의 관계가 있다고 본다. 

0. CNN
- convolution layer을 통해 feature mapping이 이루어지고 pooling을 통해 고정된 벡터로 바꿔주어, FC로의 연결이 가능하게 한다. 

1. R-CNN
- it propose regions, classify propsed regions one at a time. **output label + bounding box.** 
![image](https://user-images.githubusercontent.com/49298791/87252274-961b1e00-c4ac-11ea-94df-864380dbbde3.png)
- 결국 분류기를 실행한 region을 골라내서 classifier가 일어나게 한다는 점. 
- 이렇게 N=2000개를 골라내어 classifier를 실행하게되는데 이것을 이미지의 모든 곳을 찾아서 classify하는것보다 훨씬 적은 노력이 들어가는 형태이다. 
- (1) input image, (2) extract region proposals using selective search(이미지 속에 class가 존재하는 예상 후보 영역을 구함) 
- (3) compute CNN features(warped region으로 동일한 크기의 이미지로 변환후에 feature map을 형성), (4) classify regions(classifies like SVM)
- (5) bounding box regression (결과물이 어디에 위치해있는지까지 보정)
- ***selective search***를 사용하여 약 2000여개의 region proposal이 이루어 진다는 점, ***multi stage***로 세 단계의 학습이 이루어 진다는 점
(Conv fine tune(CNN의 feature mapping) -> SVM classifier -> BB regression)
- 하지만 합성곱 신경망 입력을 위한 고정된 크기를 위해 warping/crop을 사용해야 하며 그 과정에서 데이터 손실이 일어난다는 점. 
- 2000개의 영역마다 CNN을 적용하기에 학습 시간이 오래걸린다는 점,. 
- 학습이 여러 단계로 이루어지며 이로 인해 긴 학습 시간과 대용량 저장 공간이 요구된다는 점. 

2. SPPNet
![image](https://user-images.githubusercontent.com/49298791/87252289-b2b75600-c4ac-11ea-9a3c-2f6e3d8f0efb.png)
- input data를 conv layer을 통과하면서 feature map을 형성하고 이 feature map으로부터 region proposal이 이루어진다는 점. 
(이미지 자체를 conv network에 넣은 후에 SPP(spatial pyramid pooling) 방법을 사용하여, 고정크기 벡터로 만들어준다는점 )
- SPP를 활용하여 R-CNN의 느린 속도를 개선한다. 


3. Fast R-CNN
![image](https://user-images.githubusercontent.com/49298791/87252319-0033c300-c4ad-11ea-9d1b-2e7be30922b6.png)
- propose regions. use convolution implementation of sliding windows to classify all the proposed regions. 
- 원래의 구현은 한번에 하나의 지역을 분류해냈음. 여기서는 sliding window를 사용하여 여러개를 동시에 classify가 가능하다는 점. 
- R-CNN&SPPnet의 한계로, 학습이 multi-stage로 진행된다는 점, 학습이 많은 시간과 저장 공간을 요구한다는 점, 실제 object detection이 느리다는 점.
- Fast R-CNN을 통해서 더 좋은 성능을 획득하며, single-stage로 학습하고, 전체 네트워크 update가 가능하게 하고, 저장 공간이 필요하지 않으며 더 빠른 시간내 학습하도록 한다. 
### 학습과정
![image](https://user-images.githubusercontent.com/49298791/87252372-63255a00-c4ad-11ea-8f3d-7deb9db6a81f.png)

- input이미지와 object proposal 사용
- 이미지를 통해 conv feature map 생성
- 각 object proposal로부터 Rol pooling layer를 통해 고정된 feature vector 생성(FCN과의 연결을 위함)
- 결국 여기서의 ROI pooling layer는 Sppnet에서의 SPP layer와 동일하다고 생각하면 된다. 
- FCN을 통해 object class 를 판별(soft max classifier과 동일) / bounding box를 조절 
<br>

### ROI pooling layer?
![image](https://user-images.githubusercontent.com/49298791/87252379-80f2bf00-c4ad-11ea-969c-011764b99ff9.png)
- ROI(region of interest) 영역(=알고자하는 후보 영역)에 해당하는 부분만 max pooling을 통해 feature map으로부터 고정된 길이의 저차원 벡터로 축소 
- 각각의 ROI는 (r,c,h,w)의 튜플 형태로 이루어져 있다. 
- 결국 h*w ROI 사이즈를 작은 윈도우 사이즈로 나눈다. 
- SppNet의 spp layer의 한 pyramid level만을 사용하는 형식과 동일하다. 
<br>

### 학습을 위한 변경점
- imagenet을 사용한 pre-trained model을 사용한다. 
- 마지막의 max pooling layer가 RoI pooling layer로 대체한다
- 신경망의 마지막 fc layer와 softmax단이 두개의 output layer로 대체된다. (원래는 이미지넷 1000개 분류)
- 신경망의 입력이 이미지와 ROI를 반영할 수 있도록 반영 
(이미지로부터 구한 ROI를 둘다 입력으로 사용할 수 있도록 변경을 줌)


### detection을 위한 fine-tuning
- region-wise sampling -> hierarachical sampling : 원래는 N=128개의 이미지로부터 sampling을 진행했었음, 하지만 이미 지정한 N=2개에서 R=128개를 만들어 
학습을 진행하여 진행속도를 향상시킨다는 점. 
- single stage : 최종 classifier과 regression까지 단방향 단계로 한번에 학습이 이루어지므로 학습 과정에서 전체 network가 업데이트 가능 
<br>


### Fast R-CNN detection (결국 ROI를 얻어내어 그 곳에 대해서 적용한다는 점) 
- 실제 환경에서 보통 2000개의 ROI, 224*224 scale과 비슷하게 사용
- 각 ROI r마다 사후 class 분포 값과 bounding box 예측값을 산출한다. 
- 각 class k마다 r에 대한 Pr값인 detection confience를 부여한다. 
- 각 class별로 non-maximum supression방법을 사용하여 산출 
- SVD를 활용하여 detection을 더 빠르게 진행할 수도 있다. 


3. Faster R-CNN
- Fast R-CNN에서도 지역제한을 위한 클러스터링의 과정이 매우 느리다는 점이다. 
- 지역과 영역들을 제한하는데 분할 알고리즘(selective search)이 아니라 신경망을 사용하는 방법입니다. 
- use convolutional network to propose regions. 

4. Mask R-CNN
5. YOLO
6. YOLO v2
7. YoLo v3
- Libaray code ref : https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/