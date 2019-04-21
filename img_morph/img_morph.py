# -*- coding: utf-8 -*-
import requests
from json import JSONDecoder
import cv2
import numpy as np
from numpy import *
import math
from matplotlib import path
'''
img1_path = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image morphing/source1.png'
img2_path = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image morphing/target1.png'
'''
img1_path = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image morphing/source2.png'
img2_path = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image morphing/target2.png'
dict1={1:{"y": 132, "x":60 },2:{"y": 145, "x":112 },3:{"y": 165, "x":90 },4:{"y": 128, "x":254 },5:{"y": 140, "x":200 },6:{"y": 167, "x":218 },7:{"y": 96, "x":151 },8:{"y": 217, "x":152 },9:{"y": 244, "x":127 },10:{"y": 264, "x":158 },11:{"y": 251, "x":179 },12:{"y": 290, "x":92 },13:{"y": 291, "x":157 },14:{"y": 288, "x":222 },15:{"y": 316, "x":121 },16:{"y": 327, "x":157 },17:{"y": 319, "x":191 },18:{"y": 133, "x":28 },19:{"y": 232, "x":37 },20:{"y": 322, "x":65 },21:{"y": 375, "x":155 },22:{"y": 328, "x":234 },23:{"y": 248, "x":272 },24:{"y": 127, "x":250 }}
dict2={1:{"y": 142, "x":39 },2:{"y": 137, "x":78 },3:{"y": 153, "x":65 },4:{"y": 123, "x":233 },5:{"y": 128, "x":194 },6:{"y": 140, "x":207 },7:{"y": 106, "x":136 },8:{"y": 206, "x":140 },9:{"y": 260, "x":115 },10:{"y": 282, "x":151 },11:{"y": 254, "x":192 },12:{"y": 360, "x":91 },13:{"y": 330, "x":157 },14:{"y": 345, "x":224 },15:{"y": 395, "x":114 },16:{"y": 376, "x":161 },17:{"y": 383, "x":204 },18:{"y": 155, "x":5 },19:{"y": 269, "x":29 },20:{"y": 327, "x":69 },21:{"y": 409, "x":164 },22:{"y": 316, "x":238 },23:{"y": 256, "x":261 },24:{"y": 140, "x":247 }}

#比例因子
ALPHA = 0.9

#人脸识别相关数据 face++所需参数
def faceAPI1(file_path):
    landmark = dict1
    
    img = img_read(file_path)  
    points = []
    Width, Length, Dim = img.shape

    points_add = [(Length-1, Width-1), (math.floor(Length/2), Width-1), (0, Width-1),
                    (Length-1, math.floor(Width/2)), (0, math.floor(Width/2)),
                    (Length-1, 0), (math.floor(Length/2), 0), (0, 0)]
    #手动添加边缘的点
    for point_add in points_add:
        points.append([point_add[1],point_add[0]])

    for i, val in enumerate(list(landmark.values())):
        x = val['y']
        y = val['x']
        points.append([x,y])
    return points

def faceAPI2(file_path):
    landmark = dict2
    
    img = img_read(file_path)  
    points = []
    Width, Length, Dim = img.shape

    points_add = [(Length-1, Width-1), (math.floor(Length/2), Width-1), (0, Width-1),
                    (Length-1, math.floor(Width/2)), (0, math.floor(Width/2)),
                    (Length-1, 0), (math.floor(Length/2), 0), (0, 0)]
    #手动添加边缘的点
    for point_add in points_add:
        points.append([point_add[1],point_add[0]])

    for i, val in enumerate(list(landmark.values())):
        x = val['y']
        y = val['x']
        points.append([x,y])
    return points

def faceAPI(file_path):
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    http_url2 = 'https://api-cn.faceplusplus.com/facepp/v3/face/analyze'
    key = "RbgpeibJA9wz3csQ5CFS-uQBHMct6e6P"
    secret = "i94IXJddG57j3JMh97jnot7gwUi3v9cX"
    data = {"api_key":key, "api_secret": secret, "return_attributes": "gender,age,smiling,beauty","return_landmark" : 1}
    files = {"image_file": open(file_path, "rb")}
    response = requests.post(http_url, data=data, files=files)

    #response的内容是JSON格式
    req_con = response.content.decode('utf-8')
    #对其解码成字典格式
    req_dict = JSONDecoder().decode(req_con)
    landmark = req_dict['faces'][0]['landmark']
    
    img = img_read(file_path)  
    points = []
    Width, Length, Dim = img.shape

    points_add = [(Length-1, Width-1), (math.floor(Length/2), Width-1), (0, Width-1),
                    (Length-1, math.floor(Width/2)), (0, math.floor(Width/2)),
                    (Length-1, 0), (math.floor(Length/2), 0), (0, 0)]
    #手动添加边缘的点
    for point_add in points_add:
        points.append([point_add[1],point_add[0]])

    for i, val in enumerate(list(landmark.values())):
        x = val['y']
        y = val['x']
        points.append([x,y])
    return points

# 读取图像，解决imread不能读取中文路径的问题
def img_read(file_path):
    image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return image

#三角剖分 rect元组 存矩形
def get_triangles(rect, points):
    subbdiv = cv2.Subdiv2D(rect)
    for p in points:
        p = tuple(p)
        subbdiv.insert(p)
    triangleList = subbdiv.getTriangleList()
    #print(triangleList)
    return triangleList

#求仿射矩阵
def solve_affine_matrix(dstTri, srcTri):
    
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
'''
def solve_affine_matrix(tri1, tri2):
    S = mat([[tri1[0], tri1[2], tri1[4]],
            [tri1[1], tri1[3], tri1[5]],
            [1, 1, 1]])
    S = np.float32(S)
    T = mat([[tri2[0], tri2[2], tri2[4]],
            [tri2[1], tri2[3], tri2[5]],
            [1, 1, 1]])
    T = np.float32(T)  
    ans = np.float32(np.dot(T, S.I))
    #print(ans)
    return ans
'''

#求出所有仿射矩阵
def get_all_affine_matrix(points, trilist, pointsm):
    affineMatrix = []
    for i in range(len(trilist)):
        xy1 = (trilist[i][0], trilist[i][1])
        xy2 = (trilist[i][2], trilist[i][3])
        xy3 = (trilist[i][4], trilist[i][5])
        index = (pointsm.index(xy1), pointsm.index(xy2), pointsm.index(xy3))
        tri1 = (points[index[0]][0], points[index[0]][1], points[index[1]][0], points[index[1]][1], points[index[2]][0], points[index[2]][1])
        tri2 = trilist[i]
        #print(tri1)
        #print(tri2)
        matrix = solve_affine_matrix(tri1, tri2)
        affineMatrix.append(matrix)
    return affineMatrix

    

#求中间图的尺寸
def decide_m_size(size1, size2):
    length = size1[0] * (1 - ALPHA) + size2[0] * ALPHA
    length = math.floor(length)
    width = size1[1] * (1 - ALPHA) + size2[1] * ALPHA
    width = math.floor(width)
    ans = (length, width)
    
    return ans

#判断某个点是否在三角形当中
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

def is_in_trig(triangle, p):
    return IsInside(triangle[0], triangle[1], triangle[2], triangle[3], triangle[4], triangle[5], p[0], p[1])

#获取中间图的所有特征点的坐标
def get_imgm_positions(points1, points2):
    pointsm = []
    for i in range(len(points1)):
        x = points1[i][0] * (1-ALPHA) + points2[i][0] * ALPHA
        y = points1[i][1] * (1-ALPHA) + points2[i][1] * ALPHA
        x = math.floor(x)
        y = math.floor(y)
        pointsm.append((x, y))
    return pointsm

#三角区域转矩形
def tri_2_rec(tri):
    x_list = [tri[0], tri[2], tri[4]]
    y_list = [tri[1], tri[3], tri[5]]
    return [min(x_list), max(x_list), min(y_list), max(y_list)]


#计算中间图某个点的像素值
def get_pixel(position, src_img, matrixList, index):
    affine = matrixList[index]
    position_matrix = mat([[position[0]], [position[1]], [1]])

    #if (position[0] == 0 and position[1] == 0):
        #print(affine)

    initial_position = affine * position_matrix
    initial_position = np.float32(initial_position)
    #转为二维坐标
    raw_position = (initial_position[0][0], initial_position[1][0])
    #print(raw_position)

    x = raw_position[0]
    y = raw_position[1]

    if (x >= src_img.shape[0]):
        x = src_img.shape[0] - 1

    if (y >= src_img.shape[1]):
        y = src_img.shape[1] - 1
    #print("INIT x = %f y = %f " %(x, y))

    x = math.floor(x)
    y = math.floor(y)

    xx = x
    yy = y

    del_x = raw_position[0] - x
    del_y = raw_position[1] - y

    if (x + 1 < src_img.shape[0]):
        xx = x + 1

    if (y + 1 < src_img.shape[1]):
        yy = y + 1

    pixel = np.zeros(3)
    pixel = (1-del_x)*(1-del_y)*src_img[x][y] + del_x*(1-del_y)*src_img[xx][y] + (1-del_x)*del_y*src_img[x][yy] + del_x*del_y*src_img[xx][yy]

    #pixel = 0
    #pixel = caculate_pixel(raw_position, src_img)
    return pixel

if __name__=="__main__":
    img1 = img_read(img1_path)
    img2 = img_read(img2_path)

    img1_size = img1.shape
    img2_size = img2.shape

    src_img1 = img1
    src_img2 = img2
 
    points1 = faceAPI1(img1_path)
    points2 = faceAPI2(img2_path)
    '''
    points1 = faceAPI(img1_path)
    points2 = faceAPI(img2_path)
    #checked!!!
    '''


    
    #中间图的尺寸 以及 特征点  以及三角形列表
    #注意 此处imgm_size的横纵需要交换
    imgm_size = decide_m_size(img1_size, img2_size)
    imgm_rect = (0, 0, imgm_size[0], imgm_size[1])
    #checked!!!
    
    
    pointsm = get_imgm_positions(points1, points2)
    
    imgm_triangleList = get_triangles(imgm_rect, pointsm)


    #np.zeros((imgm_size[0]+1, imgm_size[1]+1, 4))


    
    
    #获得仿射矩阵列表
    affineMatrixList1 = get_all_affine_matrix(points1, imgm_triangleList, pointsm)
    #print(len(affineMatrixList1))
    affineMatrixList2 = get_all_affine_matrix(points2, imgm_triangleList, pointsm)
    #checked!!!
    
    img_mid = np.zeros((imgm_size[0]+1, imgm_size[1]+1, 4))

    #遍历中间图中的三角形 src_img 原图的像素分布:
    for i, index in enumerate(imgm_triangleList):
        print("i = %d" %i)
        rect = tri_2_rec(index)        
        for xx in range(int(rect[0]), int(rect[1]+1)):
            for yy in range(int(rect[2]), int(rect[3]+1)):
                if (is_in_trig(index, (xx, yy))):
                    #print("RAW: x: %d y: %d" %(xx, yy))
                    img_mid[xx][yy] = np.float32((1-ALPHA) * (get_pixel((xx, yy), src_img1, affineMatrixList1, i))) + ALPHA * (get_pixel((xx, yy), src_img2, affineMatrixList2, i))
    #print(img_mid)
    cv2.imwrite('./result.png', img_mid)
    
    


    
