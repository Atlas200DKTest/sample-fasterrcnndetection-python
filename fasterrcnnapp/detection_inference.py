# encoding=utf-8

from  ConstManager import *
import ModelManager
import hiai
from hiai.nn_tensor_lib import DataType
import os
import numpy as np
import ChannelManager
from presenter_types import *
import cv2
import time
import client


'''
fasterrcnn模型不需要对齐，输入800*600,图像输入必须处理成numpy格式
'''
class detectionInference(object):
    def __init__(self):
        self.model = ModelManager.ModelManager()
        self.width = 800
        self.height = 600
        self.clientsocket = None
        self.graph = None
        self.channel_manager=ChannelManager.ChannelManager()
        self._getgraph()

    def dispose(self):
        hiai.hiai._global_default_graph_stack.get_default_graph().destroy()

    def _getgraph(self):
        inferenceModel = hiai.AIModelDescription('faster_rcnn', detection_model_path)
        #print inferenceModel
        self.graph = self.model.CreateGraphWithoutDVPP(inferenceModel)
        if self.graph is None:
            print "CreateGraph failed"

    def Inference(self,input_image):
        inputImageTensor = hiai.NNTensor(input_image, self.width, self.height, 3, 'testImage', DataType.UINT8_T,self.width * self.height * 3)
        imageinfo=np.array([800,600,3]).astype(np.float32)
        imageinfo=np.reshape(imageinfo,(1,3))
        infoTensor = hiai.NNTensor(imageinfo, 1,3,1,'testinfo',DataType.FLOAT32_T, imageinfo.size)
        #print('inputImageTensor is :', inputImageTensor)
        datalist=[inputImageTensor, infoTensor]
        nntensorList = hiai.NNTensorList(datalist)
        #print('nntensorList', nntensorList)
        resultList = self.model.Inference(self.graph, nntensorList)
        print('inference over')
        return resultList

    def GetDetectionInfo(self,resultList):
        if not resultList:
            return  None
        tensor_num = np.reshape(resultList[0], 32)
        tensor_bbox = np.reshape(resultList[1], (64, 304, 8))
        #index = 0
        detection_reuslt = []
        for attr in range(32):
            num = int(tensor_num[attr])
            for bbox_idx in range(num):
                class_idx = attr * 2
                lt_x = tensor_bbox[class_idx][bbox_idx][0]
                lt_y = tensor_bbox[class_idx][bbox_idx][1]
                rb_x = tensor_bbox[class_idx][bbox_idx][2]
                rb_y = tensor_bbox[class_idx][bbox_idx][3]
                score = tensor_bbox[class_idx][bbox_idx][4]
                detection_reuslt.append([lt_x, lt_y, rb_x, rb_y, attr, score])
        #while True:
            #if resultList[0][index][0][0][2]>0.9:
                #print(resultList[0][index][0][0][2])
                #result = [resultList[0][index][0][0][2], resultList[0][index][0][0][3], resultList[0][index][0][0][4], resultList[0][index][0][0][5], resultList[0][index][0][0][6]]
                #detection_reuslt.append(result)
            #else:
                #break
            #index = index + 1
        return detection_reuslt

    def GetDetectionImage(self, input_image, detection_result):
        if detection_result is None:
            return None
        imageList=[]
        h,w,c = input_image.shape
        for resultList in detection_result:
            imageList.append(input_image[w*resultList[1]:w*resultList[3],h*resultList[2],h*resultList[4]])
        return imageList

    def GetImageFrameData(self, fact_info,input_image):
        image_frame = ImageFrame()
        image_frame.format = 0
        image_frame.width = input_image.shape[1]
        image_frame.height = input_image.shape[0]
	#cv2.imwrite('1-2.jpg',input_image)
        image_frame.data = cv2.imencode(".jpg",input_image)[1].tobytes()
        #with open('1-3.jpg','wb') as f:
        #    f.write(image_frame.data)
        image_frame.size = 0
        #print "fact_info in inference",fact_info
        if fact_info :
            for result in fact_info:
                dr = DetectionResult()
                dr.lt.x = (int)(result[0] * image_frame.width / 800.0)
                if dr.lt.x < 0:
                    dr.lt.x = 0
                dr.lt.y = (int)(result[1] * image_frame.height / 600.0)
                if dr.lt.y < 0:
                    dr.lt.y = 0
                dr.rb.x = (int)(result[2] * image_frame.width / 800.0)
                if dr.rb.x < 0:
                    dr.rb.x = 0
                dr.rb.y = (int)(result[3] * image_frame.height / 600.0)
                if dr.rb.y < 0:
                    dr.rb.y = 0
                dr.result_text = kind[result[4]] + ": " + str(result[5])
                image_frame.detection_results.append(dr)
                #print "result:",result
        all_data = self.channel_manager.PackRequestData(image_frame)
        return  all_data

def dowork(src_image,detection_app):
    #img = cv2.imread("./1.jpg")
    input_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (detection_app.width, detection_app.height))
    resultList = detection_app.Inference(input_image)
    #print("resulylist[0] max:",max(resultList[0][0][0][0]))
    #if resultList:
    #    print resultList[0]
    fact_info = detection_app.GetDetectionInfo(resultList)
    all_data = detection_app.GetImageFrameData(fact_info, src_image)
    if all_data:
        detection_app.clientsocket.send_data(all_data)

def sqEngine(rtsp_queue,detection_app):
    while True:
        frame = rtsp_queue.get()
        if frame is None:
            time.sleep(0.1)
            continue
        dowork(frame,detection_app)
