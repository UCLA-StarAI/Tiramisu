from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from image_synthesis.utils.misc import instantiate_from_config

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class CocoDataset(Dataset):
    def __init__(self, data_root, phase = 'train', im_preprocessor_config=None, drop_caption_rate=0.0):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.root = os.path.join(data_root, phase)
        # input_file = os.path.join(data_root, input_file)
        caption_file = "captions_"+phase+"2017.json"
        caption_file = os.path.join(data_root, "annotations", caption_file)

        self.json_file = json.load(open(caption_file, 'r'))
        print("length of the dataset is ")
        print(len(self.json_file['annotations']))

        self.num = len(self.json_file['annotations'])
        # self.image_prename = "COCO_" + phase + "2017_"
        self.image_prename = ""
        # self.folder_path = os.path.join(data_root, phase+'2017', phase+'2017')
        self.folder_path = os.path.join(data_root, phase+'2017')
 
        self.drop_rate = drop_caption_rate
        self.phase = phase
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        this_item = self.json_file['annotations'][index]
        caption = this_item['caption'].lower()
        image_name = str(this_item['image_id']).zfill(12)
        image_path = os.path.join(self.folder_path, self.image_prename+image_name+'.jpg')
        image = load_img(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']
        data = {
                'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                'text': caption if (self.phase != 'train' or self.drop_rate < 1e-6 or random.random() >= self.drop_rate) else '',
        }
        
        return data
