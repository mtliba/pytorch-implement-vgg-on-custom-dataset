import torch
import os
import datetime
import numpy as np
import cv2
from torch.utils import data
from torchvision import utils
from torchvision.datasets.vision import VisionDataset 


# The DataLoader for our specific datataset with extracted frames
class Custom(VisionDataset):

  def __init__(self, split ,resolution = (224,224) ,root, train=True, transform=None, target_transform=None, val_perc = 0.03):
        super(Custom, self).__init__(root, transform=transform,
                                target_transform=target_transform)

        self.frames_path = os.path.join(root) 
        self.number_of_classes = len(os.listdir(self.frames_path))
        self.resolution = resolution
        self.list_of_len = [len(os.path.join(self.frames_path,str(i)))) for i in range(self.number_of_classes)]
        maxx = max(list_of_len)
        # we create dictionary where key are class and value list of frames
        self.dataset_list = []
        start = datetime.datetime.now().replace(microsecond=0) # Gives accurate human readable time, rounded down not to include too many decimals
        for j in range(maxx):
            for i in range( number_of_classes+1): 
                if j == 0 :
                    one_class_frames = os.listdir(os.path.join(frames_path,str(i)))
                    print("for class {} the frames are {}".format(i, len(one_class_frames))) 
                    frame_files_sorted = sorted(one_class_frames, key = lambda x: int(x.split(".")[0]) )
                pair = (str(i),frame_files_sorted[j])
               
                self.dataset_list.append(pair)

                if i%5==0:
                    print("class {} finished.".format(i))
                    print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))
        
        # Split the dataset to validation and training
        limit = int(round(val_perc*len(self.video_list)))
        if split == "validation":
          self.dataset_list = self.dataset_list[:limit]
 
        elif split == "train":
          self.dataset_list = self.dataset_list[limit:]
        else :
            pass




  def __len__(self):
        'Denotes the total number of samples'
        return len(self.number_of_classes*sum(self.list_of_len))

  def __getitem__(self, pair_index):

        'Generates one sample of data'
        # Select sample (images,label)        frames = self.video_list[video_index]
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                         std = [0.229, 0.224, 0.225])
        calss ,frame = self.dataset_list[pair_index]
        # Load and preprocess frames
        path_to_frame = os.path.join(self.frames_path, calss, frame)

        X = cv2.imread(path_to_frame)
        if self.resolution!=None:
           X = cv2.resize(X, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
        X = X.astype(np.float32)
        X = normalize(X)

        X = torch.FloatTensor(X)
        X = X.permute(2,0,1) # swap channel dimensions

        data =X.unsqueeze(0)
        target = troch.zeros(self.number_of_classes)
        target[int(class)] = 1
        target = target.unsqueeze(0)
        if self.transform is not None:
            img = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
