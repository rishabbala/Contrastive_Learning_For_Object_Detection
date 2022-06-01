import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET
import collections

class TestDataLoader(Dataset):

    def __init__(self, folder_dir, dataset, size, transform=None):
        self.dataset = dataset
        self.resize = torchvision.transforms.Resize((size, size))
        if dataset == 'cifar10':
            self.data_dir = folder_dir
            self.labels = torch.load(self.data_dir + 'labels.pth')
            self.transform = transform
        elif dataset == 'voc':
            self.data_dir = folder_dir
            self.data = {int(i[:-4])-1: torchvision.io.read_image(self.data_dir+'/'+i).float()/255.0 for i in os.listdir(self.data_dir)}
            self.labels = {int(i[:-4])-1: ET.parse('./datasets/VOCdevkit/VOC2007/Annotations/'+i) for i in os.listdir('./datasets/VOCdevkit/VOC2007/Annotations/')}
            self.transform = transform

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def parse_voc_xml(self, node):
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def __getitem__(self, idx):
        img = self.data[idx]
        # img = self.resize(img)
        tree = self.labels[idx].getroot()
        label = self.parse_voc_xml(tree)

        return img, label