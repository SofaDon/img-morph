# -*- coding: utf-8 -*-
import requests
from json import JSONDecoder
import cv2
import numpy as np
import math

img1_path = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image morphing/source1.png'
img2_path = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image morphing/target1.png'

#人脸识别相关数据 face++所需参数
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
    data = req_dict
    #print(data)
    img = img_read(file_path)  
    vis = img.copy()
    points = []
    Length, Width, Dim = img.shape
    points_add = [(Length-1, Width-1), (math.floor(Length/2), Width-1), (0, Width-1),
                    (Length-1, math.floor(Width/2)), (0, math.floor(Width/2)),
                    (Length-1, 0), (math.floor(Length/2), 0), (0, 0)]

    for point_add in points_add:
        points.append([point_add[1],point_add[0]])
        cv2.circle(vis, (point_add[1],point_add[0]), 2, (0,0,255),-1)

    #print(len(data['faces'][0]['landmark']))
    for j in (0,len(data['faces'])-1):
        for i in data['faces'][j]['landmark']:
            cor=data['faces'][j]['landmark'][i]
            x=cor["x"]
            y=cor["y"]
            points.append([x,y])
            cv2.circle(vis, (x,y), 2, (0,0,255),-1)
    cv2.imshow("Image", vis)  
    cv2.waitKey(0)
    #print(points)


# 读取图像，解决imread不能读取中文路径的问题
def img_read(file_path):
    image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return image

#三角剖分 rect元组 存矩形
def getTriangles(rect,points):
    subbdiv = cv2.Subdiv2D(rect)
    for p in points:
        subbdiv.insert(p)
    triangleList = subbdiv.getTriangleList()
    return triangleList

if __name__=="__main__":
    img1 = img_read(img1_path)
    img2 = img_read(img2_path)
    img1_size = img1.shape
    img2_size = img2.shape
    #print(img1_size)
    faceAPI(img2_path)
    #print(img1_size)


    
