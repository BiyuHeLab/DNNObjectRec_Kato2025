# Kato M., et al., 2025, for reviewers
Analysis scripts and codes for reproducing figures in the manuscript Systematic image perturbations reveal persistent gaps between biological and machine vision

## Organization
### **codes**
Codes to reproduce the figures. To plot figures, go to the home directory and execute scripts from that directory. The reulted figures will be saved in **analysis** folder.

### **stim/**
#### **stim/codes** 
Codes used to generate the image set. The code that generates Tex images is available at [texture_synthesis](https://github.com/LefdRida/texture_synthesis). The only modification we made was target layers: change from `layers = ["conv1_1", "pool1", "pool2", "pool3", "pool4"]` to `layers = ["conv1_1","conv1_2", "pool1","conv2_1","conv2_2", "pool2"]`

#### **stim/rawim/**
Raw images taken from the validation set of the ImageNet Large Scale Visual Recognition Challenge 2012. Due to the copyright restrictions, we only include raw images used to generate exemple images shown in Fig. 1a and Fig. S1c, which do not come from the actual stimulus set used in the experiments. Nevertheless, since we provide ImageNet image IDs (`ImageNet_image_IDs.csv`) and the background masks, you will be able to reproduce all images once listed images are placed in this folder.

#### **stim/mask/**
Background masks to isolate objects. 

#### **stim/genim/**
Generated images used in the experiments. Only exemple images are included.

#### **stim/imagenet_subclass**
`Category label.csv` denotes ImageNet classes mapped onto that category, defined based on the WordNet lexical hierarchy. For exemple, when evaluating DNNs' prediction to `bear` images, the following ImageNet classes were consided as correct.

| WordNet synset ID   | ImageNet class |
| -------- | ------- |
|n02134418|	sloth_bear|
|n02134084|	ice_bear|
|n02133161|	american_black_bear|
|n02132136|	brown_bear|

`Category label_list.csv` denotes human answers that were counted as correct (the first colum) and incorrect (the second colmn).

### **RawData**
This folder includes the following, which is available at [osf](XXX)
- human responses in Exp1 and Exp2
- DNNs' top-1 predictions
- Unit activations in the DNNs' hidden layers
- IT predictability of each DNN, obtained from [Brain-score](https://www.brain-score.org/vision/leaderboard)
