##################################################
# Training Config
##################################################
GPU = '0'                   # GPU
workers = 4                 # number of Dataloader workers
epochs = 160                # number of epochs
batch_size = 64            # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (448, 448)     # size of training images
net = 'resnet101'  # feature extractor
num_attentions = 32         # number of attention maps
beta = 5e-2                 # param for update feature centers


visual_path = '/home/bhuvnesh.kumar/Downloads/Projects/CAL_Adversarial/CAL/fgvc/Bird/reversed/'

##################################################
# Dataset/Path Config
##################################################
tag = 'bird'                # 'aircraft', 'bird', 'car', or 'dog'
adv = False
# checkpoint model for resume training
import os
print(os.getcwd())
ckpt = '/home/bhuvnesh.kumar/Downloads/Projects/CAL/fgvc/FGVC/bird/Reversed/wsdan-resnet101-cal/model_bestacc.pth'