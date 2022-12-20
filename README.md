# KWU-AI-HACKATHON
2022 광운대학교 AI 해커톤 동상 솔루션입니다.
 

##사용 모델  
- Resnet50_v2  
- EfficientNet-b8  

서로 다른 activation function을 사용하는 모델을 앙상블하기 위해 Resnet과 EfficientNet을 사용했습니다.  
EfficientNet은 무겁지만 좋은 성능을 위해 b8 모델을 사용했고, Resnet은 비교적 학습 속도가 빠르고 성능도 괜찮다고 알려진 Resnet50_v2 모델을 사용했습니다.  

## 학습 방법 
- Resnet50_v2 (3 folds, epoch 40, batch size 32)  
- EfficientNet-b8 (5 folds, epoch 30, batch size 16 / 12)  
- Augmentation은 처음에 Flip과 Rotation을 사용했으나 Rotation만 사용했을 때 점수가 약간 더 높았습니다.  
- 9월 29일 이후로 본격적인 학습을 시작하게 되어 시간적 / 물리적 한계로 비교적 적은 fold와 epoch로 학습했고, EfficientNet-b8의 경우, Colab gpu 할당량을 초과하는 학습 시간이 필요해서 Check Point를 설정 했습니다.  
- EfficientNet-b8의 초기 학습은 batch size 16으로 진행했고, Check Point를 불러와서 학습할 때는 메모리 상의 문제로 batch size를 12로 수정하여 학습 시켰습니다.
- optimizer, scheduler는 baseline과 동일하고, scheduler의 경우엔 gamma를 0.9로 설정 했습니다.
- loss function은 MultiLabelSoftMarginLoss를 사용 했습니다.
- valid accuracy가 일정 횟수(코드에서는 4회)동안 증가하지 않으면 early stopping 시켰습니다.
- 후처리로 TTA(90~360도 회전)와 soft voting(threshold 0.4)을 했습니다.  

## 특이 사항  
최종 제출 코드를 재현하기 위해 알아두셔야 할 사항들 입니다.
- Check Point를 이용하는 경우, checkpoint 폴더를 추가 생성하셔야 문제없이 작동합니다.  
- Resnet50_v2 모델은 3 folds 중 fold 1, fold 2만 사용 되었습니다. (fold 0 제외)  
- EfficientNet-b8 모델은 5 folds 중 fold 0~3까지만 사용 되었습니다. (fold 4 제외)  
 
최종 제출일에 학습 도중 런타임 만료가 되는 이슈가 있어서 예정과 다르게 미완성된 모델을 사용하게 되었습니다.  
