import torch
from torch.utils.data import  Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
import torchvision

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    

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

    def __init__(self, root, annFile, transform=None, target_transform=None, category_names = None, final_img_size=128, time_step=1):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.final_img_size = final_img_size     
        self.time_step = time_step
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
        #Delete next line 
        #index = 572 + 109
        img_id = self.ids[index]
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        
        #print(img_id)
            
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img_size = img.size
        img_size_x = img_size[0]
        img_size_y = img_size[1]
        #seg_masks = torch.zeros([len(self.category),self.final_img_size,self.final_img_size])
        
        instance_types = []
        
        for i in range(len(target)):    
            instance = target[i]
            instance_types.append(instance['category_id'])

        idx_list = [i for i in range(len(instance_types)) if (instance_types[i] in self.category and len(target[i]['segmentation'])==1)]
        num_object = len(idx_list)
        
        seg_masks = torch.zeros([num_object,len(self.category),self.final_img_size,self.final_img_size])
        bboxes = torch.zeros([num_object,len(self.category),self.final_img_size,self.final_img_size])
        ins_area = torch.zeros([num_object])
        
        fg_category = 9999*torch.ones([num_object])
        for i in range(num_object):   
            idx = idx_list[np.random.choice(len(idx_list),1)[0]]
            idx_list.remove(idx)
            instance = target[idx]
            
            ins_area[i] = instance['area']/(img_size_x*img_size_y)
            
            mask = Image.new('L', (img_size_x, img_size_y))
            for j in range(len(instance['segmentation'])):
                poly = instance['segmentation'][j]
                ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
            
            mask= self.transform2(mask)
            if torch.max(mask) != 0:
                mask = mask/torch.max(mask)
                
            seg_masks[i,self.category.index(instance['category_id']),:,:] = mask.squeeze(0)
            fg_category[i] = self.category.index(instance['category_id'])
            
            bbox = instance['bbox']
            bbox_mask = Image.new('L', (img_size_x, img_size_y))
            ImageDraw.Draw(bbox_mask).rectangle([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]], outline=1, fill=1)
            bbox_mask= self.transform2(bbox_mask)
            if torch.max(bbox_mask) != 0:
                bbox_mask = bbox_mask/torch.max(bbox_mask)
            bboxes[i,self.category.index(instance['category_id']),:,:] = bbox_mask.squeeze(0)
            
        if self.transform is not None:
            img = self.transform(img)



        seg_masks = torch.clamp(seg_masks,0,1)
        bboxes = torch.clamp(bboxes,0,1)
        
        sample = {'image': img, 'seg_mask': seg_masks, 'bboxes': bboxes, 'fg_category': fg_category, 'num_object':num_object, 'ins_area':ins_area}
        return sample

    def __len__(self):
        return len(self.ids)

    def discard_small(self, min_area, max_area=1):
        #category_id = self.coco.getCatIds(catNms=category_name)
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
                if instance['category_id'] in self.category and len(instance['segmentation'])==1:
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

    def discard_num_objects(self,num_min_obj=0, num_max_obj=1):
        
        temp = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            instance_types = []
            
            for i in range(len(target)):    
                instance = target[i]
                instance_types.append(instance['category_id'])
    
            idx_list = [i for i in range(len(instance_types)) if (instance_types[i] in self.category and len(target[i]['segmentation'])==1)]
            num_object = len(idx_list)
        
            if num_object>num_min_obj and num_object <= num_max_obj:
                temp.append(img_id)
        
        print(str(len(self.ids)) + '-->' + str(len(temp)))
        self.ids = temp
        
      
#-------------------------Example-----------------------------------------
if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((128,128)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])    
    dataset = CocoData(root = 'C:/Users/motur/coco/images/train2017',
                            annFile = 'C:/Users/motur/coco/annotations/instances_train2017.json',
                            category_names =  ['giraffe','elephant','zebra','sheep','cow','bear'],
                            transform=transform, time_step = 5)
    
    #dataset.discard_small(0.01)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)   
    print('Number of samples: ', len(dataset))
    #Discarding images contain small instances  
    dataset.discard_small(min_area=0.0, max_area= 1)
    #dataset.discard_bad_examples('bad_examples_list.txt')
    #dataset.discard_num_objects()

    path1  = 'C:/Users/motur/coco/mask/bbox_sheep/'
    path2  = 'C:/Users/motur/coco/images_gt/gt_train_2/'
    
    num_object = []
    MIN_AREA = 0.01
    count = 0
    for num_iter, sample_batched in enumerate(train_loader,0):
        #image= sample_batched['image'][0]
        #imshow(torchvision.utils.make_grid(image))
        num_object.append(sample_batched['num_object'][0])
        #plt.pause(0.001)
        #y_all = sample_batched['seg_mask'][0]
        #bbox_all = sample_batched['bboxes'][0]
        #ins_area = sample_batched['ins_area'][0]
        #num_fg_obj = y_all.size()[0]
        #torchvision.utils.save_image(image, path2 +  str(count) + '.png', nrow=1, padding=0, normalize=True, range=None, scale_each=False, pad_value=0)
        count +=1        

#        imshow(torchvision.utils.make_grid(mask[0,0,:,:]))
#        plt.pause(0.001)
#        imshow(torchvision.utils.make_grid(mask[0,1,:,:]))
#        plt.pause(0.001)
#        imshow(torchvision.utils.make_grid(mask[1,0,:,:]))
#        plt.pause(0.001)
#        imshow(torchvision.utils.make_grid(mask[1,1,:,:]))
#        plt.pause(0.001)
        #print(sample_batched['num_object'][0])
        #print(sample_batched['fg_category'][0])
#        for t in range(num_fg_obj):
#            y_ = y_all[t,:,:,:]
#            y_reduced = torch.sum(y_,0).clamp(0,1).view(1,128,128)
#            bbox_ = bbox_all[t,:,:,:]
#            bbox_reduced = torch.sum(bbox_,0).clamp(0,1).view(1,128,128)
#            
#            if ins_area[t]>MIN_AREA:
#                fixed_p1 = path1 +  str(count) + '.png'
#                fixed_p2 = path2 +  str(count) + '.png'
#                count += 1
#                torchvision.utils.save_image(bbox_reduced, fixed_p1, nrow=1, padding=0, normalize=True, range=None, scale_each=False, pad_value=0)
#                torchvision.utils.save_image(y_reduced, fixed_p2, nrow=1, padding=0, normalize=True, range=None, scale_each=False, pad_value=0)
                
    
    num_object = np.array(num_object)
    print(np.max(num_object))
    print(np.mean(num_object))
    print(np.median(num_object))
    plt.hist(num_object)
    plt.xticks(range(1, 22))
    plt.show()
    plt.savefig('num_obj.png')
    
    
