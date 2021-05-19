import os
from xml.dom import minidom
import numpy as np
import cv2

# path='./VOC2007/VOCdevkit/VOC2007/Annotations'
# path1='./VOC2007/VOCdevkit/VOC2007/JPEGImages'
# path='./VOC2007/test/VOCdevkit/VOC2007/Annotations'
# path1='./VOC2007/test/VOCdevkit/VOC2007/JPEGImages'
def get_marklabel():
    t = os.listdir('./utils/label')
    label = []
    labelcode=[]
    count = 0
    for i in t:
        label.append([i.split('_')[0]])
        labelcode.append(count)
        count += 1
    return label,labelcode
def getFileNames(path):
    filenames = os.listdir(path)
    xmlfile=[]
    count=0
    for i, filename in enumerate(filenames):
            count+=1
            sFileName=filename
            xmlfile.append(sFileName)
    return xmlfile,count


# 返回该xml文件下对于的jpg图片，
#     filepath=path1+'\\'+file         #图片路径
#     objectname = []                  #目标
#     xmin = []                        #目标的定点x
#     ymin = []
#     xmax = []
#     ymax = []
#     lgname                        #目标个数
def return_data(path,path1):
    dom=minidom.parse(path)
    file=dom.getElementsByTagName('filename')[0].firstChild.data
    filepath=path1+'/'+file         #图片路径
    objectname = []                  #目标
    xmin = []                        #目标的定点x
    ymin = []
    xmax = []
    ymax = []
    lgname=0                        #目标个数
    s1=dom.getElementsByTagName('name')
    lgname=len(s1)-1
    for i in range(0,lgname):
        objectname.append(s1[i+1].firstChild.data)
    s1=dom.getElementsByTagName('xmin')
    for i in range(0,len(s1)):
        uu=float(s1[i].firstChild.data)
        xmin.append(int(uu))
    s1 = dom.getElementsByTagName('ymin')
    for i in range(0,len(s1)):
        uu=float(s1[i].firstChild.data)
        ymin.append(int(uu))
    s1 = dom.getElementsByTagName('xmax')
    for i in range(0,len(s1)):
         uu=float(s1[i].firstChild.data)
         xmax.append(int(uu))
    s1 = dom.getElementsByTagName('ymax')
    for i in range(0,len(s1)):
         uu = float(s1[i].firstChild.data)
         ymax.append(int(uu))
    position=[]
    for i in range(0,lgname):
        position.append([ymin[i],xmin[i],ymax[i],xmax[i]])

    return filepath,objectname,position,lgname

def return_data1(path,path1):
    label,labelcode=get_marklabel()
    label=np.array(label)
    labelcode=np.array(labelcode)
    dom=minidom.parse(path)
    file=dom.getElementsByTagName('filename')[0].firstChild.data
    filepath=path1+'/'+file         #图片路径
    objectname = []                  #目标
    xmin = []                        #目标的定点x
    ymin = []
    xmax = []
    ymax = []

    s1=dom.getElementsByTagName('name')
    lgname=len(s1)-1 #目标个数
    for i in range(0,lgname):
        objectname.append(s1[i+1].firstChild.data)
    s1=dom.getElementsByTagName('xmin')
    for i in range(0,len(s1)):
        uu=float(s1[i].firstChild.data)
        xmin.append(int(uu))
    s1 = dom.getElementsByTagName('ymin')
    for i in range(0,len(s1)):
        uu=float(s1[i].firstChild.data)
        ymin.append(int(uu))
    s1 = dom.getElementsByTagName('xmax')
    for i in range(0,len(s1)):
         uu=float(s1[i].firstChild.data)
         xmax.append(int(uu))
    s1 = dom.getElementsByTagName('ymax')
    for i in range(0,len(s1)):
         uu = float(s1[i].firstChild.data)
         ymax.append(int(uu))
    position=[]
    for i in range(0,lgname):
        position.append([ymin[i],xmin[i],ymax[i],xmax[i]])
    newposition=[]
    newobjectname=[]
    newlgname=0
    cls_sort=[]
    for i in range(0,len(objectname)):
        if(objectname[i] in label):
            cls=np.where(label==objectname[i])

            cls_sort.append(list(labelcode[cls[0]]))
            newlgname+=1
            newposition.append(position[i])
            newobjectname.append(objectname[i])

    return filepath,newobjectname,newposition,cls_sort,newlgname
#提取目标
# path  xml文件的路径

def getdataxy1(path,path1,startcount,endcount,xmlfile):
     ms=[]
     for j in range(startcount,endcount):
        p = path + '/' + xmlfile[j]
        filepath, objectname, po, lgname = return_data(p,path1)
        ms.append([filepath,objectname,po,lgname])
     return ms


def getdataxy2(path, path1, startcount, endcount, xmlfile):
    ms = []
    for j in range(startcount, endcount):
        p = path + '/' + xmlfile[j]
        filepath,newobjectname,newposition,cls_sort,newlgname= return_data1(p, path1)
        if(newlgname>0):
              ms.append([filepath,newobjectname,newposition,cls_sort,newlgname])
    return ms
# from utils import config
# configg=config.Config()
# xmlfile, count = getFileNames(configg.path)  # 得到该路径下有哪些xml文件
# print(count)
# # ms=getdataxy2(configg.path,configg.path1,0,5000,xmlfile)
# # for k in ms:
# #     print(k)