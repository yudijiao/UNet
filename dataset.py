import torch.utils.data as data
import os
import PIL.Image as Image

#data.Dataset:
#所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)

class LiverDataset(data.Dataset):
    #创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self,root,transform = None,target_transform = None):#root表示图片路径
        n = len(os.listdir(root))//2 #os.listdir(path)返回指定路径下的文件和文件夹列表。/是真除法,//对结果取整
        
        imgs = []
        for i in range(n):
            img = os.path.join(root,"%03d.png"%i)#os.path.join(path1[,path2[,......]]):将多个路径组合后返回
            mask = os.path.join(root,"%03d_mask.png"%i)
            imgs.append([img,mask])#append只能有一个参数，加上[]变成一个list
        
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    
    def __getitem__(self,index):
        x_path,y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x,img_y#返回的是图片
    
    
    def __len__(self):
        return len(self.imgs)#400,list[i]有两个元素，[img,mask]
