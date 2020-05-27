import torch
from torch.utils.data import  Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
import torchvision
from utils import imshow
import cv2

class CocoData(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        category_names : name of the categories desired dataset consists
        final_img_size : Dataset image size, default: 128
        
        
        Return: 
            'image'  : 3x128x128
            'segmentation mask' : num_catx128x128  --- only one  instance for specific category (one instance for each category)
            'category' : multiple categories (e.g. zebra, giraffe)
        
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, category_names = None, final_img_size=128):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.final_img_size = final_img_size     
        self.transform2 = transforms.Compose([
                                               transforms.Scale((final_img_size,final_img_size)),
                                               transforms.ToTensor(),
                                           ])
    
        
        if category_names == None:
            self.category = None
            self.ids = list(self.coco.imgs.keys())
        else:
            self.category = self.coco.getCatIds(catNms=category_names) #e.g. [22,25]
            
            self.ids = []
            self.cat = []
            for x in self.category:
                self.ids +=  self.coco.getImgIds(catIds=x )
                self.cat +=  [x]*len(self.coco.getImgIds(catIds=x )) #e.g. [22,22,...,22]

            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        valid_instance = False
        num_iter = 0
        while valid_instance == False and (num_iter < 5):
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            img_size = img.size
            img_size_x = img_size[0]
            img_size_y = img_size[1]
            seg_masks = torch.zeros([len(self.category),self.final_img_size,self.final_img_size])
            single_fg_mask = torch.zeros([len(self.category),self.final_img_size,self.final_img_size])
            instance_types = []
            
            for i in range(len(target)):    
                instance = target[i]
                instance_types.append(instance['category_id'])
    
            idx_list = [i for i in range(len(instance_types)) if (instance_types[i] in self.category and len(target[i]['segmentation'])>=1)]
            num_object = len(idx_list)

            for i in range(num_object):   
                idx = idx_list[np.random.choice(len(idx_list),1)[0]]
                idx_list.remove(idx)
                instance = target[idx]
    
                mask = Image.new('L', (img_size_x, img_size_y))
                for j in range(len(instance['segmentation'])):
                    poly = instance['segmentation'][j]
                    ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
                
                if i==0:
                    bbox = instance['bbox']
                    mask_instance_1 = mask
                    bbox_mask = Image.new('L', (img_size_x, img_size_y))
                    ImageDraw.Draw(bbox_mask).rectangle([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]], outline=1, fill=1)
                    bbox_mask= self.transform2(bbox_mask)
                    if torch.max(bbox_mask) != 0:
                        bbox_mask = bbox_mask/torch.max(bbox_mask)
                    
                
                mask= self.transform2(mask)
                if torch.max(mask) != 0:
                    mask = mask/torch.max(mask)
    
                seg_masks[self.category.index(instance['category_id']),:,:] += mask.squeeze()
                
                #Single foreground object mask
                if i==0:
                    single_fg_obj_ctg = self.category.index(instance['category_id'])
                    single_fg_mask[single_fg_obj_ctg,:,:] = mask
                                       
            
            if self.transform is not None:
                img = self.transform(img)
    
    
            seg_masks = torch.clamp(seg_masks,0,1)
            
            
            ###bounding-box of the object in resized image
            if bbox[2]>bbox[3]:
                dx = 0
                dy = (bbox[2]-bbox[3]) /2
            else:
                dx = (bbox[3]-bbox[2]) /2
                dy = 0
                
            x1 = max(0,bbox[0]-dx)
            y1 = max(0,bbox[1]-dy)
            x2 = min(img_size_x,bbox[0]+bbox[2]+dx)
            y2 = min(img_size_y,bbox[1]+bbox[3]+dy)
            
            mask_instance_1 = mask_instance_1.crop((int(x1), int(y1), int(x2), int(y2) ))
            mask_instance_1 = self.transform2(mask_instance_1)
            if torch.max(mask_instance_1) != 0:
                    mask_instance_1 = mask_instance_1/torch.max(mask_instance_1)
            
            mask_instance = torch.zeros([len(self.category),self.final_img_size,self.final_img_size])
            mask_instance[single_fg_obj_ctg,:,:] = mask_instance_1  
            
            x_scale = self.final_img_size/img_size_x
            y_scale = self.final_img_size/img_size_y           
            x1, x2 = x1*x_scale, x2*x_scale
            y1, y2 = y1*y_scale, y2*y_scale
            
            #bbox_scaled = [y1,y2,x1,x2]
            bbox_scaled = [int(y1),int(y2),int(x1),int(x2)]
            
            num_iter += 1
            if (bbox_scaled[1]>bbox_scaled[0]) and (bbox_scaled[3]>bbox_scaled[2]):
                valid_instance = True

        if not (bbox_scaled[1]>bbox_scaled[0]):
            if bbox_scaled[1]<self.final_img_size:
                bbox_scaled[1] += 1
            else:
                bbox_scaled[0] -= 1
        if not (bbox_scaled[3]>bbox_scaled[2]):
            if bbox_scaled[3]<self.final_img_size:
                bbox_scaled[3] += 1
            else:
                bbox_scaled[2] -= 1
                
        
        sample = {'image': img, 'seg_mask': seg_masks,
                  'single_fg_mask': single_fg_mask,
                  'mask_instance':mask_instance, 'bbox':bbox_scaled,
                  'cat': self.cat[index], 'num_object':num_object}
        return sample

    def __len__(self):
        return len(self.ids)

    def discard_small(self, min_area, max_area=1):
        temp = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            instance_types = []
            valid_mask = False
            
            path = self.coco.loadImgs(img_id)[0]['file_name']
            img = Image.open(os.path.join(self.root, path))
            img_size = img.size
            img_size_x = img_size[0]
            img_size_y = img_size[1]

            total_fg_area = 0
            total_fg_area_relevant = 0
            
            for i in range(len(target)):    
                instance = target[i]
                total_fg_area += instance['area']
                instance_types.append(instance['category_id'])
                if instance['category_id'] in self.category and len(instance['segmentation'])>=1:
                    total_fg_area_relevant +=  instance['area']
                    valid_mask = True
                if (instance['category_id'] in self.category) and (type(instance['segmentation']) is not list):
                    valid_mask = False
                    break 

            if valid_mask and total_fg_area_relevant/(img_size_x*img_size_y) > min_area and total_fg_area/(img_size_x*img_size_y) < max_area:
                temp.append(img_id)
        
        print(str(len(self.ids)) + '-->' + str(len(temp)))
        self.ids = temp
        

    def discard_bad_examples(self, path):  
        file_list = open(path, "r")
        bad_examples = file_list.readlines()
        for i in range(len(bad_examples)):
            bad_examples[i] = int(bad_examples[i][:-1])

        temp = []
        for img_id in self.ids:
            if not (img_id in bad_examples):
                temp.append(img_id)
        
        print(str(len(self.ids)) + '-->' + str(len(temp)))
        self.ids = temp
        print('Bad examples are left out!')        


#-------------------------Example-----------------------------------------


class cityscape(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        category_names : name of the categories desired dataset consists
        final_img_size : Dataset image size, default: 128


        Return:
            'image'  : 3x128x128
            'segmentation mask' : num_catx128x128  --- only one  instance for specific category (one instance for each category)
            'category' : multiple categories (e.g. zebra, giraffe)

    """

    def __init__(self, imfile, mfiles, transform=None, target_transform=None, category_names=None, final_img_size=256):
        self.imfile  = imfile
        self.mfiles = mfiles
        self.transform = transform
        self.target_transform = target_transform
        self.final_img_size = final_img_size
        self.transform2 = transforms.Compose([
            transforms.Scale((final_img_size, final_img_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """

        valid_instance = False
        num_iter = 0
        while valid_instance == False and (num_iter < 5):
            mf = self.mfiles[index]
            mask = np.array(Image.open(mf).convert('L')) / 255.
            box = self.find_bbx(mask)
            filename = "_".join(os.path.basename(mf).split('_')[:-1]) + '.png'
            mode = mf.split('/pedestrian/')[0].split('/')[-1]
            f = os.path.join(self.imfile, mode + '_img', filename)

            img = np.array(Image.open(f).convert('RGB'))

            img = img[box[0]:box[2], box[1]:box[3], :]
            mask = mask[box[0]:box[2], box[1]:box[3]]

            img_size = img.shape
            img_size_x = img_size[1]
            img_size_y = img_size[0]
            seg_masks = torch.zeros([1, self.final_img_size, self.final_img_size])
            single_fg_mask = torch.zeros([1, self.final_img_size, self.final_img_size])
            instance_types = []
            num_object = 1
            bbox = np.array([0, 0, 0, 0])
            for i in range(num_object):
                if i == 0:
                    xs = np.nonzero(np.sum(mask, axis=0))[0]
                    ys = np.nonzero(np.sum(mask, axis=1))[0]
                    bbox[0] = xs.min()
                    bbox[2] = xs.max() - xs.min()
                    bbox[1] = ys.min()
                    bbox[3] = ys.max() - ys.min()
                    mask_instance_1 = mask
                    bbox_mask = Image.new('L', (img_size_x, img_size_y))
                    ImageDraw.Draw(bbox_mask).rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                                                        outline=1, fill=1)
                    bbox_mask = self.transform2(bbox_mask)
                    if torch.max(bbox_mask) != 0:
                        bbox_mask = bbox_mask / torch.max(bbox_mask)

                mask = self.transform2(Image.fromarray(mask))
                if torch.max(mask) != 0:
                    mask = mask / torch.max(mask)

                seg_masks[0, :, :] += mask.squeeze()

                # Single foreground object mask
                if i == 0:
                    single_fg_obj_ctg = 0
                    single_fg_mask[single_fg_obj_ctg, :, :] = mask

            if self.transform is not None:
                img = self.transform(Image.fromarray(img))

            seg_masks = torch.clamp(seg_masks, 0, 1)

            ###bounding-box of the object in resized image
            if bbox[2] > bbox[3]:
                dx = 0
                dy = (bbox[2] - bbox[3]) / 2
            else:
                dx = (bbox[3] - bbox[2]) / 2
                dy = 0

            x1 = max(0, bbox[0] - dx)
            y1 = max(0, bbox[1] - dy)
            x2 = min(img_size_x, bbox[0] + bbox[2] + dx)
            y2 = min(img_size_y, bbox[1] + bbox[3] + dy)
            
            mask_instance_1 = Image.fromarray(mask_instance_1)
            mask_instance_1 = mask_instance_1.crop((int(x1), int(y1), int(x2), int(y2)))
            mask_instance_1 = self.transform2(mask_instance_1)
            if torch.max(mask_instance_1) != 0:
                mask_instance_1 = mask_instance_1 / torch.max(mask_instance_1)

            mask_instance = torch.zeros([1, self.final_img_size, self.final_img_size])
            mask_instance[single_fg_obj_ctg, :, :] = mask_instance_1

            x_scale = self.final_img_size / img_size_x
            y_scale = self.final_img_size / img_size_y
            x1, x2 = x1 * x_scale, x2 * x_scale
            y1, y2 = y1 * y_scale, y2 * y_scale

            # bbox_scaled = [y1,y2,x1,x2]
            bbox_scaled = [int(y1), int(y2), int(x1), int(x2)]

            num_iter += 1
            if (bbox_scaled[1] > bbox_scaled[0]) and (bbox_scaled[3] > bbox_scaled[2]):
                valid_instance = True

        if not (bbox_scaled[1] > bbox_scaled[0]):
            if bbox_scaled[1] < self.final_img_size:
                bbox_scaled[1] += 1
            else:
                bbox_scaled[0] -= 1
        if not (bbox_scaled[3] > bbox_scaled[2]):
            if bbox_scaled[3] < self.final_img_size:
                bbox_scaled[3] += 1
            else:
                bbox_scaled[2] -= 1

        sample = {'image': img, 'seg_mask': seg_masks,
                  'single_fg_mask': single_fg_mask,
                  'mask_instance': mask_instance, 'bbox': bbox_scaled,
                  'cat': 0, 'num_object': num_object}
        return sample

    def find_bbx(self, maskj):
        maskj = np.expand_dims(maskj, axis=-1)

        box = np.array([0, 0, 0, 0])

        # Compute Bbx coordinates

        margin = 3
        xs = np.nonzero(np.sum(maskj, axis=0))[0]
        ys = np.nonzero(np.sum(maskj, axis=1))[0]
        box[1] = xs.min() - margin
        box[3] = xs.max() + margin
        box[0] = ys.min() - margin
        box[2] = ys.max() + margin

        if box[0] < 0: box[0] = 0
        if box[1] < 0: box[1] = 0

        h = box[2] - box[0]
        w = box[3] - box[1]
        if h < w:
            diff = w - h
            half = int(diff / 2)
            box[0] -= half
            if box[0] < 0:
                box[2] -= box[0]
                box[0] = 0
            else:
                box[2] += diff - half

            if box[2] > maskj.shape[0]:
                box[2] = maskj.shape[0]
        else:
            diff = h - w
            half = int(diff / 2)
            box[1] -= half
            if box[1] < 0:
                box[3] -= box[1]
                box[1] = 0
            else:
                box[3] += diff - half
            if box[3] > maskj.shape[1]:
                box[3] = maskj.shape[1]

        # if box[3] > resized_height: box[3] = resized_height - 1
        # if box[2] > resized_width: box[2] = resized_width - 1

        if box[3] == box[1]:
            box[3] += 1
        if box[0] == box[2]:
            box[2] += 1

        # bbx[box[0]:box[2], box[1]:box[3], :] = 1

        return box

    def __len__(self):
        return len(self.mfiles)

    # -------------------------Example-----------------------------------------


class MHP(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        category_names : name of the categories desired dataset consists
        final_img_size : Dataset image size, default: 128


        Return:
            'image'  : 3x128x128
            'segmentation mask' : num_catx128x128  --- only one  instance for specific category (one instance for each category)
            'category' : multiple categories (e.g. zebra, giraffe)

    """

    def __init__(self, imfile, mfiles, transform=None, target_transform=None, category_names=None, final_img_size=256):
        self.imfile = imfile
        self.mfiles = mfiles
        self.transform = transform
        self.target_transform = target_transform
        self.final_img_size = final_img_size
        self.transform2 = transforms.Compose([
            transforms.Scale((final_img_size, final_img_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """

        valid_instance = False
        num_iter = 0
        while valid_instance == False and (num_iter < 5):
            mf = self.mfiles[index]
            mask = np.array(cv2.imread(mf))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = (mask > 0) * 1.
            f = os.path.join(self.imfile, os.path.basename(mf).split('_')[0] + '.jpg')
            img = np.array(Image.open(f).convert('RGB'))
            img_size = img.shape
            img_size_x = img_size[1]
            img_size_y = img_size[0]
            seg_masks = torch.zeros([1, self.final_img_size, self.final_img_size])
            single_fg_mask = torch.zeros([1, self.final_img_size, self.final_img_size])
            instance_types = []
            num_object = 1
            bbox = np.array([0, 0, 0, 0])
            for i in range(num_object):
                if i == 0:
                    xs = np.nonzero(np.sum(mask, axis=0))[0]
                    ys = np.nonzero(np.sum(mask, axis=1))[0]
                    bbox[0] = xs.min()
                    bbox[2] = xs.max() - xs.min()
                    bbox[1] = ys.min()
                    bbox[3] = ys.max() - ys.min()
                    mask_instance_1 = mask
                    bbox_mask = Image.new('L', (img_size_x, img_size_y))
                    ImageDraw.Draw(bbox_mask).rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                                                        outline=1, fill=1)
                    bbox_mask = self.transform2(bbox_mask)
                    if torch.max(bbox_mask) != 0:
                        bbox_mask = bbox_mask / torch.max(bbox_mask)

                mask = self.transform2(Image.fromarray(mask))
                if torch.max(mask) != 0:
                    mask = mask / torch.max(mask)

                seg_masks[0, :, :] += mask.squeeze()

                # Single foreground object mask
                if i == 0:
                    single_fg_obj_ctg = 0
                    single_fg_mask[single_fg_obj_ctg, :, :] = mask

            if self.transform is not None:
                img = self.transform(Image.fromarray(img))

            seg_masks = torch.clamp(seg_masks, 0, 1)

            ###bounding-box of the object in resized image
            if bbox[2] > bbox[3]:
                dx = 0
                dy = (bbox[2] - bbox[3]) / 2
            else:
                dx = (bbox[3] - bbox[2]) / 2
                dy = 0

            x1 = max(0, bbox[0] - dx)
            y1 = max(0, bbox[1] - dy)
            x2 = min(img_size_x, bbox[0] + bbox[2] + dx)
            y2 = min(img_size_y, bbox[1] + bbox[3] + dy)

            mask_instance_1 = Image.fromarray(mask_instance_1)
            mask_instance_1 = mask_instance_1.crop((int(x1), int(y1), int(x2), int(y2)))
            mask_instance_1 = self.transform2(mask_instance_1)
            if torch.max(mask_instance_1) != 0:
                mask_instance_1 = mask_instance_1 / torch.max(mask_instance_1)

            mask_instance = torch.zeros([1, self.final_img_size, self.final_img_size])
            mask_instance[single_fg_obj_ctg, :, :] = mask_instance_1

            x_scale = self.final_img_size / img_size_x
            y_scale = self.final_img_size / img_size_y
            x1, x2 = x1 * x_scale, x2 * x_scale
            y1, y2 = y1 * y_scale, y2 * y_scale

            # bbox_scaled = [y1,y2,x1,x2]
            bbox_scaled = [int(y1), int(y2), int(x1), int(x2)]

            num_iter += 1
            if (bbox_scaled[1] > bbox_scaled[0]) and (bbox_scaled[3] > bbox_scaled[2]):
                valid_instance = True

        if not (bbox_scaled[1] > bbox_scaled[0]):
            if bbox_scaled[1] < self.final_img_size:
                bbox_scaled[1] += 1
            else:
                bbox_scaled[0] -= 1
        if not (bbox_scaled[3] > bbox_scaled[2]):
            if bbox_scaled[3] < self.final_img_size:
                bbox_scaled[3] += 1
            else:
                bbox_scaled[2] -= 1

        sample = {'image': img, 'seg_mask': seg_masks,
                  'single_fg_mask': single_fg_mask,
                  'mask_instance': mask_instance, 'bbox': bbox_scaled,
                  'cat': 0, 'num_object': num_object}
        return sample

    def __len__(self):
        return len(self.mfiles)

    # -------------------------Example-----------------------------------------

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((128,128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])    
    dataset = CocoData(root = 'C:/Users/motur/coco/images/train2017',
                            annFile = 'C:/Users/motur/coco/annotations/instances_train2017.json',
                            category_names = ['giraffe'],
                            transform=transform)
    
    dataset.discard_small(0.03)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)   
    print('Number of samples: ', len(dataset))
    
  
    for num_iter, sample_batched in enumerate(train_loader,0):
        image= sample_batched['image'][0]
        imshow(torchvision.utils.make_grid(image))
        plt.pause(0.001)
        mask= sample_batched['seg_mask'][0]
        fg_mask= sample_batched['single_fg_mask'][0]
        imshow(torchvision.utils.make_grid(mask[0]))
        plt.pause(0.001)
     