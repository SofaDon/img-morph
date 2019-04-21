# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import requests
from json import JSONDecoder
import time
import cv2
from numpy import *
import numpy as np
import random

img1_path = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image morphing/source1.png'
img2_path = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image morphing/target1.png'
#比例因子
ALPHA = 0.5

#人脸识别相关数据 face++所需参数
def faceAPI(file_path):
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    http_url2 = 'https://api-cn.faceplusplus.com/facepp/v3/face/analyze'
    key = "RbgpeibJA9wz3csQ5CFS-uQBHMct6e6P"
    secret = "i94IXJddG57j3JMh97jnot7gwUi3v9cX"
    data = {"api_key":key, "api_secret": secret, "return_attributes": "gender,age,smiling,beauty","return_landmark" : 2}
    files = {"image_file": open(img1_path, "rb")}
    response = requests.post(http_url, data=data, files=files)
    #response的内容是JSON格式
    req_con = response.content.decode('utf-8')
    #对其解码成字典格式
    req_dict = JSONDecoder().decode(req_con)
    landmark = req_dict['faces'][0]['landmark']
    
    img = img_read(file_path)  
    points = []
    Width, Length, Dim = img.shape
    print(img.shape)
    points_add = [(Length-1, Width-1), (math.floor(Length/2), Width-1), (0, Width-1),
                    (Length-1, math.floor(Width/2)), (0, math.floor(Width/2)),
                    (Length-1, 0), (math.floor(Length/2), 0), (0, 0)]
    #手动添加边缘的点
    for point_add in points_add:
        points.append([point_add[1],point_add[0]])

    for i, val in enumerate(list(landmark.values())):
        x = val['x']
        y = val['y']
        points.append([x,y])
    return points

# 读取图像，解决imread不能读取中文路径的问题
def img_read(file_path):
    image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return image

#三角剖分
def getTriangles(rect,points):
    subbdiv = cv2.Subdiv2D(rect)
    for p in points:
        subbdiv.insert(p)
    triangleList = subbdiv.getTriangleList()
    return triangleList

#求解仿射矩阵
def getAffineTransform(srcTri, dstTri):
    
    #Tri为点的list
    srcMat = mat([[srcTri[0], srcTri[2], srcTri[4]],
                    [srcTri[1], srcTri[3], srcTri[5]],
                    [1, 1, 1]])
    srcMat = np.float32(srcMat)
    dstMat = mat([[dstTri[0], dstTri[2], dstTri[4]],
                    [dstTri[1], dstTri[3], dstTri[5]],
                    [1, 1, 1]])
    dstMat = np.float32(dstMat)
    #print(srcMat)
    #print(dstMat)
    transMat = np.float32(dstMat * srcMat.I)
    #print(transMat)
    return transMat


#生成中间图片特征点
def getMiddlePoints(s_points, t_points):
    m_points = []
    for i in range(len(s_points)):
        x = (1-a) * s_points[i][0] + a * t_points[i][0]
        y = (1-a) * s_points[i][1] + a * t_points[i][1]
        x = int(math.floor(x))
        y = int(math.floor(y))
        m_points.append((x,y))
    return m_points

#判断点是否在三角形中
def IsTrangleOrArea(x1,y1,x2,y2,x3,y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
 
def IsInside(x1,y1,x2,y2,x3,y3,x,y):
 
    #三角形ABC的面积
    ABC = IsTrangleOrArea(x1,y1,x2,y2,x3,y3)
 
    # 三角形PBC的面积
    PBC = IsTrangleOrArea(x,y,x2,y2,x3,y3)
 
    # 三角形ABC的面积
    PAC = IsTrangleOrArea(x1,y1,x,y,x3,y3)
 
    # 三角形ABC的面积
    PAB = IsTrangleOrArea(x1,y1,x2,y2,x,y)
 
    return (ABC == PBC + PAC + PAB)

def ifInTriangle(p, triangle):
    return IsInside(triangle[0], triangle[1], triangle[2], triangle[3], triangle[4], triangle[5], p[0], p[1])

#得到三角形对应的矩形区域
def tri2rec(triangle):
    x_list = [triangle[0], triangle[2], triangle[4]]
    y_list = [triangle[1], triangle[3], triangle[5]]
    return [min(x_list), max(x_list), min(y_list), max(y_list)]

def bilinear(s_img, p):
    #print("bilinear")
    #print(s_img.shape)
    
    i = math.floor(p[0])
    j = math.floor(p[1])
    u = p[0] - i
    v = p[1] - j
    '''
    j = math.floor(p[0])
    i = math.floor(p[1])
    v = float(p[0]) - j
    u = float(p[1]) - i
    '''
    #print(s_img.shape)
    #print(i,j,1-u,1-v)
    if i + 1 < s_img.shape[1]:
        s = i + 1
    else:
        s = s_img.shape[1] - 1
    if j + 1 < s_img.shape[0]:
        t = j + 1
    else:
        t = s_img.shape[0] - 1
    if i >= s_img.shape[1]:
        i = s_img.shape[1] - 1
    if j >= s_img.shape[0]:
        j = s_img.shape[0] - 1
    
    #print(s_img[j][i])
    m_img = (1-u)*(1-v)*s_img[j][i] + (1-u)*v*s_img[t][i] + u*(1-v)*s_img[j][s] + u*v*s_img[t][s]
    #print(m_img)
    return m_img

if __name__ == '__main__':
    s_points = faceAPI(img1_path)
    t_points = faceAPI(img2_path)
    #print(s_points)
    #print(t_points)
    s_img = img_read(img1_path)
    t_img = img_read(img2_path)
    s_size = s_img.shape
    t_size = t_img.shape
    
    s_rect = (0, 0, s_size[1], s_size[0])
    t_rect = (0, 0, t_size[1], t_size[0])
    s_triangleList = getTriangles(s_rect, s_points)
    t_triangleList = getTriangles(t_rect, t_points)
    #print(s_triangleList)
    #print(t_triangleList)
    getAffineTransform(s_triangleList[1],t_triangleList[1])

    #得到中间图大小
    m_x = (1 - a) * s_size[1] + a * t_size[1]
    m_x = math.floor(m_x)
    m_y = (1 - a) * s_size[0] + a * t_size[0]
    m_y = math.floor(m_y)
    m_rect = (0, 0, m_x, m_y)
    #print(m_rect)
    #构造中间图
    #global m_img
    m_result = np.ones((m_y + 1, m_x + 1, 4))
    #遍历中间图所有三角形
    m_points = getMiddlePoints(s_points, t_points)
    print(m_points)
    m_triangleList = getTriangles(m_rect, m_points)

    for t in range(len(m_triangleList)):
    
        #得到坐标范围
        tri = m_triangleList[t]
        #print(tri)
        m_index = []
        ii = 0
        for m in range(3):
            m_index.append(m_points.index((int(tri[ii]),int(tri[ii+1]))))
            ii += 2
        coor_range = tri2rec(tri)
        #print("coor_range")
        #print(coor_range)
        count1 = 0
        count2 = 0
        for i in range(int(coor_range[0]), int(coor_range[1]) + 1):
            for j in range(int(coor_range[2]), int(coor_range[3]) + 1):
                if ifInTriangle([i, j], tri): 
                    #得到对应的仿射矩阵
                    s_tri = []
                    t_tri = []
                    for n in range(3):
                        s_tri.append(s_points[m_index[n]][0])
                        s_tri.append(s_points[m_index[n]][1])
                        t_tri.append(t_points[m_index[n]][0])
                        t_tri.append(t_points[m_index[n]][1])
                    transMat2src = getAffineTransform(tri,s_tri)
                    transMat2dst = getAffineTransform(tri,t_tri)
                    #得到对应的点
                    m2s = transMat2src * mat([i, j, 1]).T
                    m2s = np.float32(m2s)
                    m2d = transMat2dst * mat([i, j, 1]).T
                    m2d = np.float32(m2d)
                    #双线性插值
                    #print('zuobiao %d %d' % (i,j))
                    #print(m_img[j][i])
                    
                    m_result[j][i] = np.float32((1 - a) * bilinear(s_img, m2s) + a * bilinear(t_img, m2d))
                    #print(j, i)
                    #print(m_result[j][i])
                    #print(m_result)
        #print(count1)
        #print(count2)
    cv2.imwrite('result-0.9.jpg',m_result)        


