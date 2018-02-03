
import numpy as np;
import xml.dom.minidom;
import cv2;
import os;
from config import *



dictionarys={'aeroplane':0,'bicycle':1,'bird':2,'boat':3,'bottle':4,
             'bus':5,'car':6,'cat':7,'chair':8,'cow':9,
             'diningtable':10,'dog':11,'horse':12,'motorbike':13,'person':14,
             'pottedplant':15,'sheep':16,'sofa':17,'train':18,'tvmonitor':19}

def getImageInfoByXML(xml_file_path):
    DomTree = xml.dom.minidom.parse(xml_file_path)
    annotation = DomTree.documentElement;
    filenamelist = annotation.getElementsByTagName('filename')
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    imgsizelist=annotation.getElementsByTagName('size')
    widthlist=imgsizelist[0].getElementsByTagName('width')
    width=widthlist[0].childNodes[0].data
    heightlist = imgsizelist[0].getElementsByTagName('height')
    height = heightlist[0].childNodes[0].data
    depthlist = imgsizelist[0].getElementsByTagName('depth')
    depth = depthlist[0].childNodes[0].data
    imgshape=[width,height,depth]
    objnames = []
    objbounds=[]
    for object in objectlist:
        objnamelist=object.getElementsByTagName('name')
        objname = objnamelist[0].childNodes[0].data
        objnames.append(objname);
        boundarray=[]
        bndboxlist=object.getElementsByTagName('bndbox')
        xminlist=bndboxlist[0].getElementsByTagName('xmin')
        xmin=xminlist[0].childNodes[0].data
        boundarray.append(xmin);
        yminlist = bndboxlist[0].getElementsByTagName('ymin')
        ymin = yminlist[0].childNodes[0].data
        boundarray.append(ymin);
        xmaxlist = bndboxlist[0].getElementsByTagName('xmax')
        xmax = xmaxlist[0].childNodes[0].data
        boundarray.append(xmax);
        ymaxlist = bndboxlist[0].getElementsByTagName('ymax')
        ymax = ymaxlist[0].childNodes[0].data
        boundarray.append(ymax);
        objbounds.append(boundarray)
    return filename,imgshape,objnames,objbounds;

def splitTrainAndValid(xml_file):
    imagelist = os.listdir(xml_file);
    shuffle_index = [i for i in range(len(imagelist))]
    #np.random.shuffle(shuffle_index);
    images=[imagelist[index] for index in shuffle_index]
    ratio=0.9;
    train_num=int(0.9*len(images));
    train_set=images[:train_num];
    valid_set=images[train_num:len(images)];
    return train_set,valid_set

def generateOneSampleByXml(ImgPath,xml_file_path):
    filename, imgshape, objnames, objbounds=getImageInfoByXML(xml_file_path);
    image_path=ImgPath+filename;
    img=cv2.imread(image_path,0);
    init_img=img;

    cell_num_h=CELL_SIZE;
    cell_num_w=CELL_SIZE;
    sample_y=np.array(np.zeros([cell_num_h,cell_num_w,25]),np.float32);
    img_w=int(imgshape[0])
    img_h=int(imgshape[1])
    target_w=IMAGE_SIZE
    target_h=IMAGE_SIZE
    img=cv2.resize(img,(target_w, target_h), interpolation=cv2.INTER_NEAREST)
    img = np.reshape(img, (target_h,target_w,1))
    spreadratio_w=target_w/img_w;
    spreadratio_h=target_h/img_h;
    cell_step_w = target_w // cell_num_w
    cell_step_h = target_h // cell_num_h
    for index in range(len(objnames)):
        objname=objnames[index]
        objboubd=objbounds[index]
        xmin=int(objboubd[0])
        ymin=int(objboubd[1])
        xmax=int(objboubd[2])
        ymax=int(objboubd[3])
        centerx=(xmin+xmax)*spreadratio_w/2
        cols=int(centerx//cell_step_w)
        centery=(ymin+ymax)*spreadratio_h/2
        rows=int(centery//cell_step_h)
        anchorw=(xmax-xmin)*spreadratio_w
        anchorh=(ymax-ymin)*spreadratio_h
        temp_index=int(rows*cell_num_w+cols)
        sample_y[rows,cols,0] = 1;
        sample_y[rows,cols,1] = centerx;
        sample_y[rows,cols,2] = centery;
        sample_y[rows,cols,3] = anchorw;
        sample_y[rows,cols,4] = anchorh;
        objname_value=dictionarys[objname]
        sample_y[rows,cols,5+objname_value]=1
    return img,sample_y;





def generateFinalDataSet(ImgPath,AnnoPath,xml_names_set,batch_size):
    sample_sum=len(xml_names_set)
    batch_num = sample_sum // batch_size
    index=0
    for i in range(batch_num):
        batch_set=xml_names_set[index:index+batch_size]
        train_x = [];
        train_y = [];
        for j in range(len(batch_set)):
            xml_file_path = AnnoPath + batch_set[j]
            img, sample_y = generateOneSampleByXml(ImgPath, xml_file_path)
            train_x.append(img)
            train_y.append(sample_y)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        yield train_x,train_y
        index = index+batch_size








