#!/usr/bin/env python
# coding: utf-8

# ## 광운대학교 AI 해커톤  
# 
# 안녕하세요 저희는 이번 광운대학교 AI 해커톤에 참가한 'n을뒤집으면u' 팀입니다.  
# 이번 기회에 처음으로 Image data를 다루게 되어 우여곡절을 겪으며 많은 것을 배울 수 있었습니다.  
# 이번 대회 다들 정말 고생 많으셨고, 아직 많이 부족하지만 저희 팀 코드를 공유합니다!  
# 
# ---------------------------------------------------------------------------  
# 저희 팀은 Colab Pro 버전을 사용하였습니다.  
# 전체적인 코드는 baseline과 크게 다르지 않습니다.  
# 
# ## 참고 자료  
# 이전에 유사한 데이터로 진행된 대회 코드들을 참고 했습니다.  
# [EfficientNet-b8, TTA 참고 자료](https://dacon.io/competitions/official/235697/codeshare/2429?page=1&dtype=recent)  
# [CheckPoint 참고 자료](https://dacon.io/competitions/official/235697/codeshare/2441?page=1&dtype=recent)  
# [Loss Function 참고 자료](https://dacon.io/competitions/official/235697/codeshare/2354?page=1&dtype=recent)  
# [Swish activation function 참고 자료_1](https://dacon.io/competitions/official/235697/codeshare/2445?page=1&dtype=recent)  
# [Swish activation function 참고 자료_2](https://blog.ceshine.net/post/pytorch-memory-swish/)  
# [Augmentation 참고 자료](https://www.dacon.io/competitions/official/235697/codeshare/2437?page=2&dtype=recent)  
# 
# ## 사용 모델  
# - Resnet50_v2  
# - EfficientNet-b8  
# 
# 서로 다른 activation function을 사용하는 모델을 앙상블하기 위해 Resnet과 EfficientNet을 사용했습니다.  
# EfficientNet은 무겁지만 좋은 성능을 위해 b8 모델을 사용했고, Resnet은 비교적 학습 속도가 빠르고 성능도 괜찮다고 알려진 Resnet50_v2 모델을 사용했습니다.  
# 
# ## 학습 방법  
# - Resnet50_v2 (3 folds, epoch 40, batch size 32)  
# - EfficientNet-b8 (5 folds, epoch 30, batch size 16 / 12)  
# - Augmentation은 처음에 Flip과 Rotation을 사용했으나 Rotation만 사용했을 때 점수가 약간 더 높았습니다.  
# - 9월 29일 이후로 본격적인 학습을 시작하게 되어 시간적 / 물리적 한계로 비교적 적은 fold와 epoch로 학습했고, EfficientNet-b8의 경우, Colab gpu 할당량을 초과하는 학습 시간이 필요해서 Check Point를 설정 했습니다.  
# - EfficientNet-b8의 초기 학습은 batch size 16으로 진행했고, Check Point를 불러와서 학습할 때는 메모리 상의 문제로 batch size를 12로 수정하여 학습 시켰습니다.
# - optimizer, scheduler는 baseline과 동일하고, scheduler의 경우엔 gamma를 0.9로 설정 했습니다.
# - loss function은 MultiLabelSoftMarginLoss를 사용 했습니다.
# - valid accuracy가 일정 횟수(코드에서는 4회)동안 증가하지 않으면 early stopping 시켰습니다.
# - 후처리로 TTA(90~360도 회전)와 soft voting(threshold 0.4)을 했습니다.  
# 
# ## 특이 사항  
# 최종 제출 코드를 재현하기 위해 알아두셔야 할 사항들 입니다.
# - Check Point를 이용하는 경우, checkpoint 폴더를 추가 생성하셔야 문제없이 작동합니다.  
# - Resnet50_v2 모델은 3 folds 중 fold 1, fold 2만 사용 되었습니다. (fold 0 제외)  
# - EfficientNet-b8 모델은 5 folds 중 fold 0~3까지만 사용 되었습니다. (fold 4 제외)  
#  
# 최종 제출일에 학습 도중 런타임 만료가 되는 이슈가 있어서 예정과 다르게 미완성된 모델을 사용하게 되었습니다.  

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# ## Data Load

# In[ ]:


from google.colab import output
# !cp 파일1 파일2 # 파일1을 파일2로 복사 붙여넣기
get_ipython().system('cp "/content/drive/MyDrive/광운대 해커톤/235697_제 2회 컴퓨터 비전 학습 경진대회_data.zip" "data_2.zip"')
# data_2.zip을 현재 디렉터리에 압축해제
get_ipython().system('unzip "data_2.zip"')


# In[3]:


from google.colab import output
# 현재 디렉터리에 dirty_mnist라는 폴더 생성
get_ipython().system('mkdir "./dirty_mnist"')
#dirty_mnist.zip라는 zip파일을 dirty_mnist라는 폴더에 압축 풀기
get_ipython().system('unzip "dirty_mnist_2nd.zip" -d "./dirty_mnist/"')
# 현재 디렉터리에 test_dirty_mnist라는 폴더 생성
get_ipython().system('mkdir "./test_dirty_mnist"')
#test_dirty_mnist.zip라는 zip파일을 test_dirty_mnist라는 폴더에 압축 풀기
get_ipython().system('unzip "test_dirty_mnist_2nd.zip" -d "./test_dirty_mnist/"')
# 출력 결과 지우기
output.clear()


# ## Library Import

# In[ ]:


get_ipython().system('pip3 install efficientnet_pytorch ttach')


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import imutils
import zipfile
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from google.colab import output
import albumentations
import ttach as tta

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 디바이스 설정


# ## Random Seed Setting

# In[6]:


import random

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)


# ## Dataset Compose

# In[7]:


dirty_mnist_answer = pd.read_csv("dirty_mnist_2nd_answer.csv")
# dirty_mnist라는 디렉터리 속에 들어있는 파일들의 이름을 
# namelist라는 변수에 저장
namelist = os.listdir('./dirty_mnist/')

# numpy를 tensor로 변환하는 ToTensor 정의
class ToTensor(object):
    """numpy array를 tensor(torch)로 변환합니다."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.FloatTensor(image),
                'label': torch.FloatTensor(label)}
# to_tensor 선언
to_tensor = T.Compose([
                        ToTensor()
                    ])
    



class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self,
                 dir_path,
                 meta_df,
                 transforms=to_tensor,#미리 선언한 to_tensor를 transforms로 받음
                 augmentations=None):
        
        self.dir_path = dir_path # 데이터의 이미지가 저장된 디렉터리 경로
        self.meta_df = meta_df # 데이터의 인덱스와 정답지가 들어있는 DataFrame

        self.transforms = transforms# Transform
        self.augmentations = augmentations # Augmentation
        
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, index):
        # 폴더 경로 + 이미지 이름 + .png => 파일의 경로
        # 참고) "12".zfill(5) => 000012
        #       "146".zfill(5) => 000145
        # cv2.IMREAD_GRAYSCALE : png파일을 채널이 1개인 GRAYSCALE로 읽음
        image = cv2.imread(self.dir_path +                           str(self.meta_df.iloc[index,0]).zfill(5) + '.png',
                           cv2.IMREAD_GRAYSCALE)
        
        # 0 ~ 255의 값을 갖고 크기가 (256,256)인 numpy array를
        # 0 ~ 1 사이의 실수를 갖고 크기가 (256,256,1)인 numpy array로 변환
        image = (image/255).astype('float')[..., np.newaxis]
        

        # 정답 numpy array생성(존재하면 1 없으면 0)
        label = self.meta_df.iloc[index, 1:].values.astype('float')
        sample = {'image': image, 'label': label}

        # transform 적용
        # numpy to tensor
        if self.transforms:
            sample = self.transforms(sample)
        # augmentation
        if self.augmentations:
          
          sample['image']= self.augmentations(sample['image'])
        

        # sample 반환
        return sample


# ## Model  
# Resnet50 모델의 weight는 imagenet1k_v2를 사용했으며, overfit 방지를 위해 dropout layer를 추가했습니다.  
# 학습 과정에서 loss function으로 MultiLabelSoftMarginLoss을 사용하기 때문에 fc layer에 sigmoid를 적용하지 않았습니다.

# In[8]:



from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# nn.Module을 상속 받아 Resnet50_v2를 정의
class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.conv2d = nn.Conv2d(1, 3, 3, stride=1)
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.dropout = nn.Dropout(p=0.2)
        self.FC = nn.Linear(1000, 26)


    def forward(self, x):
        
        # resnet을 추가
        x = self.conv2d(x)
        x = F.relu(self.resnet(x))
        x = self.dropout(x)
        x = self.FC(x)
        return x


# In[9]:


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Efficientnet(nn.Module):
    def __init__(self):
        super(Efficientnet, self).__init__()
        self.conv2d = nn.Conv2d(1, 3, 3, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.swish = MemoryEfficientSwish()
        self.FC = nn.Linear(1000, 26)
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b8',advprop=True)


    def forward(self, x):
        x = self.conv2d(x)
        # efficientnet을 추가
        x = self.efficientnet(x)
        x = self.dropout(x)
        x = self.swish(x)
        x = self.FC(x)
        return x


# ## Check Point

# In[10]:


#체크포인트 파일 생성

PATH_CP = "/content/drive/MyDrive/광운대 해커톤/checkpoint/checkp.pt"

if not os.path.exists(PATH_CP):
    with open(PATH_CP, 'w'): 
        pass


# In[ ]:


#checkpoint file이 비어있는지 확인
os.path.getsize(PATH_CP) ==0


# ## Train  
# 저희 팀은 장시간 코드를 돌리다가 여러 우여곡절을 겪었습니다.  
# 대부분 이미 알고 계실테지만 혹시라도 필요하신 분들을 위해 [colab run time 유지 방법](https://www.dacon.io/forum/405904)을 첨부합니다.  
# 
# 

# In[ ]:



# cross validation을 적용하기 위해 KFold 생성
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# dirty_mnist_answer에서 train_idx와 val_idx를 생성
folds=[]
for fold_index, (trn_idx, val_idx) in enumerate(kf.split(dirty_mnist_answer),1):
    folds.append((trn_idx, val_idx))

checkpoint =0
epoch_cp = 0
valid_loss = 0
fold_cp = 0
num_epoch = 40
for fold in range(3):
    print(f'[folds: {fold}]')
    # cuda cache 초기화
    torch.cuda.empty_cache()

    #fold별로 train_idx와 val_idx 설정 
    trn_idx = folds[fold][0]
    val_idx = folds[fold][1]

    #train fold, validation fold 분할
    train_answer = dirty_mnist_answer.iloc[trn_idx]
    test_answer  = dirty_mnist_answer.iloc[val_idx]

    #적용할 augmentation 설정
    train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(180),
            T.ToTensor()
        ])
    
    valid_transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor()
        ])
    
    #Dataset 정의
    train_dataset = DatasetMNIST("dirty_mnist/", train_answer, augmentations=train_transform)
    valid_dataset = DatasetMNIST("dirty_mnist/", test_answer, augmentations=valid_transform)


    #DataLoader 정의
    train_data_loader = DataLoader(
        train_dataset,
        batch_size = 32,
        shuffle = True,
        num_workers=4
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size = 32,
        shuffle = False,
        num_workers=4
    )

    # 모델 선언
    model = Resnet50()
    #model = Efficientnet()
    model.to(device)# gpu에 모델 할당

    # 훈련 옵션 설정
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = 1e-3)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size = 5,
                                                gamma = 0.9)
    
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    
    #체크포인트 불러오기
    if os.path.getsize(PATH_CP)==0:
      pass
    else:
      if epoch_cp != -1:
          
          PATH_CP = "/content/drive/MyDrive/광운대 해커톤/checkpoint/checkp.pt"

          checkpoint = torch.load(PATH_CP)
          model.load_state_dict(checkpoint['model_state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          epoch_cp = checkpoint['epoch']
          valid_loss = checkpoint['loss']
          fold_cp = checkpoint['fold']
          model.eval()

      #체크포인트 fold로 넘기기
      if fold_cp > fold:
          continue
    
    # 훈련 시작
    valid_acc_max = 0
    early_stop_count = 0
    for epoch in range(num_epoch):
        #check point 이후부터 학습
        if os.path.getsize(PATH_CP)==0:
          pass 
        elif epoch_cp+1 == num_epoch:
          break
        elif epoch_cp+1 > epoch:
          continue
        # 1개 epoch 훈련
        train_acc_list = []
        with tqdm(train_data_loader,#train_data_loader를 iterative하게 반환
                total=train_data_loader.__len__(), # train_data_loader의 크기
                unit="batch") as train_bar:# 한번 반환하는 smaple의 단위는 "batch"
            for sample in train_bar:
                train_bar.set_description(f"Train Epoch {epoch}")
                # 갱신할 변수들에 대한 모든 변화도를 0으로 초기화
                # 참고)https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html
                optimizer.zero_grad()
                images, labels = sample['image'], sample['label']
                # tensor를 gpu에 올리기 
                labels = labels.to(device)
                images = images.to(device)




                # 모델의 dropoupt, batchnormalization를 train 모드로 설정
                model.train()
                # .forward()에서 중간 노드의 gradient를 계산
                with torch.set_grad_enabled(True):
                    # 모델 예측
                    probs  = model(images)
                    # loss 계산
                    loss = criterion(probs, labels)
                    # 중간 노드의 gradient로
                    # backpropagation을 적용하여
                    # gradient 계산
                    loss.backward()
                    # weight 갱신
                    optimizer.step()

                    # train accuracy 계산
                    probs  = probs.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    preds = probs > 0.5
                    batch_acc = (labels == preds).mean()
                    train_acc_list.append(batch_acc)
                    train_acc = np.mean(train_acc_list)

                # 현재 progress bar에 현재 미니배치의 loss 결과 출력
                train_bar.set_postfix(train_loss= loss.item(),
                                      train_acc = train_acc)
                

        # 1개 epoch학습 후 Validation 점수 계산
        valid_acc_list = []
        valid_loss_list = []

        with tqdm(valid_data_loader,
                total=valid_data_loader.__len__(),
                unit="batch") as valid_bar:
            for sample in valid_bar:
                valid_bar.set_description(f"Valid Epoch {epoch}")
                optimizer.zero_grad()
                images, labels = sample['image'], sample['label']
                labels = labels.to(device)
                images = images.to(device)

                # 모델의 dropoupt, batchnormalization를 eval모드로 설정
                model.eval()
                # .forward()에서 중간 노드의 gradient를 계산
                with torch.no_grad():
                    # validation loss만을 계산
                    probs  = model(images)
                    valid_loss = criterion(probs, labels)

                    # train accuracy 계산
                    probs  = probs.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    preds = probs > 0.5
                    batch_acc = (labels == preds).mean()
                    valid_acc_list.append(batch_acc)

                valid_acc = np.mean(valid_acc_list)
                valid_loss_list.append(valid_loss.item())
                valid_loss = np.mean(valid_loss_list)
                valid_bar.set_postfix(valid_loss = valid_loss,
                                      valid_acc = valid_acc)
                
            
        # Learning rate 조절
        lr_scheduler.step()

        # 모델 저장
        if valid_acc_max < valid_acc:
            early_stop_count = 0
            valid_acc_max = valid_acc
            best_model = model
            #MODEL = "EfficientNet_b8"
            MODEL = "Resnet50_v2"
            # 모델을 저장할 구글 드라이브 경로
            path = "/content/drive/MyDrive/광운대 해커톤/models/"
            torch.save(best_model, f'{path}{fold}_{MODEL}_{valid_loss.item():2.4f}_epoch_{epoch}.pth')

            
            #이전보다 valid_acc가 높을 때 체크포인트 저장
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss
                }, PATH_CP)
            
        else:
          early_stop_count+=1
        if early_stop_count > 3:
           print('early stop')
           break

    epoch_cp = -1

    # 폴드별로 가장 좋은 모델 저장
    torch.save(best_model, f'{path}best_model_epoch_{epoch}.pth')


# ## Test Data & Models Load

# In[13]:


#test Dataset 정의
sample_submission = pd.read_csv("sample_submission.csv")
test_dataset = DatasetMNIST("test_dirty_mnist/", sample_submission)
batch_size = 16
test_data_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 3,
    drop_last = False
)


# In[14]:


best_models=[]
path = "/content/drive/MyDrive/광운대 해커톤/best models"
model1 = torch.load(path + "/0_EfficientNet_b8_0.1328_epoch_28.pth")
model2 = torch.load(path + "/1_EfficientNet_b8_0.1579_epoch_21.pth")
model3 = torch.load(path + "/2_EfficientNet_b8_0.1589_epoch_26.pth")
model4 = torch.load(path + "/3_EfficientNet_b8_0.1670_epoch_24.pth")
model5 = torch.load(path + "/1_Resnet50_v2_0.2059_epoch_39.pth")
model6 = torch.load(path + "/2_Resnet50_v2_0.2125_epoch_39.pth")

best_models.append(model1)
best_models.append(model2)
best_models.append(model3)
best_models.append(model4)
best_models.append(model5)
best_models.append(model6)


# ## TTA

# In[15]:


import ttach as tta

transformer = tta.Compose(
    [
        tta.Rotate90(angles=[0,90,180,270]),
    ]
)


# ## Ensemble / Predict

# In[ ]:


predictions_list = []
# 배치 단위로 추론
prediction_df = pd.read_csv("sample_submission.csv")

for model in best_models:
    # 0으로 채워진 array 생성
    prediction_array = np.zeros([prediction_df.shape[0],
                                 prediction_df.shape[1] -1])
    

    for idx, sample in enumerate(test_data_loader):
        with torch.no_grad():
            # 추론
            model.eval()
            tta_model = tta.ClassificationTTAWrapper(model,transformer,merge_mode='mean')
            images = sample['image']
            images = images.to(device)
            probs  = tta_model(images)
            probs = probs.detach().cpu().numpy()

            # 예측 결과를 
            # prediction_array에 입력
            batch_index = batch_size * idx
            prediction_array[batch_index: batch_index + images.shape[0],:]                         = probs.astype(np.float32)
                         
    # 채널을 하나 추가하여 list에 append
    predictions_list.append(prediction_array[...,np.newaxis])


# In[ ]:


# axis = 2를 기준으로 평균
predictions_array = np.concatenate(predictions_list, axis = 2)
predictions_mean = predictions_array.mean(axis = 2)

#threshold 0.4로 설정
predictions_mean = (predictions_mean > 0.4) * 1
predictions_mean


# ## 제출파일 생성

# In[ ]:


sample_submission = pd.read_csv("sample_submission.csv")
sample_submission.iloc[:,1:] = predictions_mean
sample_submission.to_csv("final_efficientb8_resnet50_v2_5.csv", index = False)
sample_submission

