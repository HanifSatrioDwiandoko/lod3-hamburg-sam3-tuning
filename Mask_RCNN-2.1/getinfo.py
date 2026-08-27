# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:48:37 2020

@author: wyf-laptop
"""

import cv2
import numpy as np
import imutils


def get_shape(im,label):
    ret, thresh = cv2.threshold(im, 228, 255, cv2.THRESH_BINARY) 

    contours = cv2.findContours(thresh,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) 
    cnts = contours[1] if imutils.is_cv2() else contours[0]  #用imutils来判断是opencv是2还是2+
    
    shape=[]
    for cnt in cnts:
        # 外接矩形框，没有方向角
        # x, y, w, h = cv2.boundingRect(cnt)
        # shape.append([[(x,y),w,h],label])
        # cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #duobianxing 
        arclen = cv2.arcLength(cnt, True)
        epsilon = max(3, int(arclen * 0.02))
        approx = cv2.approxPolyDP(cnt, epsilon, False) 
        x_poly=[]
        y_poly=[]
        for i in range(len(approx)):
            x_poly.append(approx[i][0][0])
            y_poly.append(approx[i][0][1])
        
        # cv2.polylines(im, [approx], True, (0, 255, 0), 1)
        polygon=[x_poly,y_poly]
        shape.append([polygon,label])
    # shape={'polygon':polygon,'label':label}
    
    return shape

def region(filename):
    im = cv2.imread(filename)
    print (im.shape)
    # imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    shapeall=[]
    # classname={'window':[],'balcony':[],'roof':[],'door':[]}
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)) # xingtaixue yunsuan he
    
    # extract roof according to HSV of blue
    # rlow=np.array([100,43,46])
    # rhig=np.array([125,255,255])
    # roof=cv2.inRange(hsv,rlow,rhig)
    # if roof.any()!=0:
    #     shape=get_shape(roof,'roof')
    #     shapeall+=shape
    # cv2.imwrite('./rr.jpg',roof)
    
    # extract window according to HSV of red1 and red2
    wlow1=np.array([156,43,46])
    whig1=np.array([180,255,255])
    window1=cv2.inRange(hsv,wlow1,whig1)
    
    wlow2=np.array([0,43,46])
    whig2=np.array([11,255,255])
    window2=cv2.inRange(hsv,wlow2,whig2)
    window=window1+window2
    window1 = cv2.erode(window,kernel)        #腐蚀图像
    windowresult = cv2.dilate(window1,kernel)      #膨胀图像
    if windowresult.any()!=0:
        shape=get_shape(windowresult,'window')
        shapeall+=shape
    # cv2.imwrite('./ww.jpg',window)
    # cv2.imwrite('./wwre.jpg',windowresult)
    # cv2.imwrite('./wwguass.jpg',window_gauss)
    
    blow=np.array([125,43,46])
    bhig=np.array([155,255,255])
    balcony=cv2.inRange(hsv,blow,bhig)
    balcony1 = cv2.erode(balcony,kernel)        #腐蚀图像
    balconyresult = cv2.dilate(balcony1,kernel)      #膨胀图像
    if balconyresult.any()!=0:
        shape=get_shape(balconyresult,'balcony')
        shapeall+=shape
    # cv2.imwrite('./bb.jpg',balony)
    
    dlow=np.array([11,43,46])
    dhig=np.array([26,255,255])
    door=cv2.inRange(hsv,dlow,dhig)
    if door.any()!=0:
        shape=get_shape(door,'door')
        shapeall+=shape
    
    return shapeall

# import os
# for filename in os.listdir(r"./"):              #listdir的参数是文件夹的路径
#     # filename.split('.')
#     print ( filename.split('.'))   
# im = r'100.png'

# shape=region(im)



# #%%
# im = cv2.imread('100.png')
# # imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

# hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)) # xingtaixue yunsuan he

# # extract roof according to HSV of blue
# rlow=np.array([100,43,46])
# rhig=np.array([125,255,255])
# roof=cv2.inRange(hsv,rlow,rhig)
# # cv2.imwrite('./rr.jpg',roof)

# # extract window according to HSV of red1 and red2
# wlow1=np.array([156,43,46])
# whig1=np.array([180,255,255])
# window1=cv2.inRange(hsv,wlow1,whig1)

# wlow2=np.array([0,43,46])
# whig2=np.array([11,255,255])
# window2=cv2.inRange(hsv,wlow2,whig2)
# window=window1+window2
# # windowresult=skimage.morphology.erosion(window,skimage.morphology.disk(3))
# # windowresult=skimage.morphology.opening(windowresult,skimage.morphology.disk(3))

# window1 = cv2.erode(window,kernel)        #腐蚀图像
# windowresult = cv2.dilate(window1,kernel)      #膨胀图像

# # window_gauss = cv2.GaussianBlur(window,(3,3),0) 
# cv2.imwrite('./ww.jpg',window)
# cv2.imwrite('./wwre.jpg',windowresult)
# # cv2.imwrite('./wwguass.jpg',window_gauss)

# blow=np.array([125,43,46])
# bhig=np.array([155,255,255])
# balony=cv2.inRange(hsv,blow,bhig)
# balony1 = cv2.erode(balony,kernel)        #腐蚀图像
# balonyresult = cv2.dilate(balony1,kernel)      #膨胀图像
# # cv2.imwrite('./bb.jpg',balony)

# dlow=np.array([11,43,46])
# dhig=np.array([26,255,255])
# door=cv2.inRange(hsv,dlow,dhig)

        
# ret, thresh = cv2.threshold(balonyresult, 228, 255, cv2.THRESH_BINARY) 

# contours = cv2.findContours(thresh,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  #cv2.RETR_EXTERNAL 定义只检测外围轮廓
#   #line, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
# cnts = contours[1] if imutils.is_cv2() else contours[0]  #用imutils来判断是opencv是2还是2+
 
# for cnt in cnts:
#     # 外接矩形框，没有方向角
#     # x, y, w, h = cv2.boundingRect(cnt)
#     # cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # 最小外接矩形框，有方向角
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
#     box = np.int0(box)
#     cv2.drawContours(im, [box], 0, (0, 0, 0), 2)
    
#     #duobianxing 
#     arclen = cv2.arcLength(cnt, True)
#     epsilon = max(3, int(arclen * 0.02))
#     approx = cv2.approxPolyDP(cnt, epsilon, False) 
#     cv2.polylines(im, [approx], True, (0, 255, 0), 1)
#     # 最小外接圆
#     # (x, y), radius = cv2.minEnclosingCircle(cnt)
#     # center = (int(x), int(y))
#     # radius = int(radius)
#     # cv2.circle(im, center, radius, (255, 0, 0), 2)
 
#     # 椭圆拟合
#     # ellipse = cv2.fitEllipse(cnt)
#     # cv2.ellipse(im, ellipse, (255, 255, 0), 2)
 
#     # 直线拟合
#     # rows, cols = im.shape[:2]
#     # [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
#     # lefty = int((-x * vy / vx) + y)
#     # righty = int(((cols - x) * vy / vx) + y)
#     # im = cv2.line(im, (cols - 1, righty), (0, lefty), (0, 255, 255), 2)
 
# # cv2.imshow('a',im)
# cv2.imwrite('./result.jpg',im)
# # cv2.waitKey(0)