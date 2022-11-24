import os
import sys
import copy
import cv2
import math
from PyQt5 import QtWidgets, QtGui, QtCore
from hw2_ui import Ui_MainWindow
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import decomposition
from torchvision import transforms
class myMainWindow(QtWidgets.QMainWindow , Ui_MainWindow):
    def __init__(self):
        super(QtWidgets.QMainWindow , self).__init__()
        self.setupUi(self)
        self.onBindingUI()
        self.imgs = []

    def onBindingUI(self):
        self.pushButton.clicked.connect(self.loadVideo)
        self.pushButton_2.clicked.connect(self.loadImage)
        self.pushButton_3.clicked.connect(self.loadFolder)
        self.pushButton_4.clicked.connect(self.backgroundSubtraction)
        self.pushButton_5.clicked.connect(self.preprocessing)
        self.pushButton_6.clicked.connect(self.videoTracking)
        self.pushButton_7.clicked.connect(self.perspectiveTransform)
        self.pushButton_8.clicked.connect(self.imageReconstruction)
        self.pushButton_9.clicked.connect(self.computeTheReconstructionError)
    def loadVideo(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'folder')
        self.video = cv2.VideoCapture(path)
    def loadImage(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'folder')
        self.img = cv2.imread(path)
    def loadFolder(self):
        self.folderPath = QtWidgets.QFileDialog.getExistingDirectory(self, 'folder')
        dir = os.listdir(self.folderPath)
        for i , image in enumerate(os.listdir(self.folderPath) , start=1):
            path = 'sample ({}).jpg'.format(i)
            img = cv2.imread(self.folderPath + '/' + path)
            self.imgs.append(img)
    def backgroundSubtraction(self):
        fgbg = cv2.createBackgroundSubtractorMOG2(history=24 , varThreshold=40)
        while(1):
            ret, frame = self.video.read()
            if ret == True:
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
                fgmask = fgbg.apply(gray)

                bmask = cv2.bitwise_and(frame ,frame ,None,mask=fgmask)
                cv2.imshow('origin',frame)
                cv2.imshow('frontground',fgmask)
                cv2.imshow('remove background' ,bmask)
                k = cv2.waitKey(30)
                if k == 27:
                    break
            else:
                break
        self.video.release()
        cv2.destroyAllWindows()
    def preprocessing(self):
        b,firstFrame = self.video.read()
        firstFrameGray = cv2.cvtColor(firstFrame , cv2.COLOR_BGR2GRAY)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.filterByArea = True
        params.minArea = 35
        params.maxArea = 90
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(firstFrameGray)
        self.pts = cv2.KeyPoint_convert(keypoints)
        self.pts = np.delete(self.pts , 5 , 0)
        for point in self.pts:
            
           
            cv2.rectangle(firstFrame,(int(point[0]) - 6, int(point[1]) - 6), (int(point[0]) + 6, int(point[1]) + 6),(0,0,255), 1)
            cv2.line(firstFrame,(int(point[0]) , int(point[1]) - 6),(int(point[0]), int(point[1]) + 6),(0,0,255), 1)
            cv2.line(firstFrame,(int(point[0]) - 6 , int(point[1])),(int(point[0]) + 6, int(point[1])),(0,0,255), 1)
        
        cv2.imshow('first frame' , firstFrame)
        cv2.waitKey(0)

    def videoTracking(self):
        
        
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (10,10),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Take first frame and find corners in it
        ret, old_frame = self.video.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        p0 = self.pts[:, np.newaxis, :]
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        while(1):
            ret,frame = self.video.read()
            if ret == False:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), [0, 0, 255], 2)
                frame = cv2.circle(frame,(int(a),int(b)),5,[0, 0, 255],-1)
            img = cv2.add(frame,mask)
            cv2.imshow('Tracking whole video',img)
            k = cv2.waitKey(30)
            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)

        cv2.destroyAllWindows()
        self.video.release()
    def perspectiveTransform(self):
        while(1):
            ret , frame = self.video.read()
            if(ret == False):
                break
            grayFrame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
            dic = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            parameters =  cv2.aruco.DetectorParameters_create()
            corners , ids , reject =  cv2.aruco.detectMarkers(grayFrame , dic , parameters = parameters) #四個角 順時針
            
            points = [] #左上角開始順時針
            ids = ids.squeeze()  
            for i in range(ids.size):
                if(ids[i] == 1):
                    topLeft = corners[i].squeeze()[0]
                if(ids[i] == 2):
                    topRight = corners[i].squeeze()[1]
                if(ids[i] == 3):
                    bottomRight = corners[i].squeeze()[2]
                if(ids[i] == 4):
                    bottomLeft = corners[i].squeeze()[3]
            points.append(topLeft)
            points.append(topRight)
            points.append(bottomRight)
            points.append(bottomLeft)
            pts = np.array(points)

            h , w , dim = self.img.shape
            img_pts = np.array([[0,0],
                                [w-1,0],
                                [w-1,h-1],
                                [0,h-1]],dtype=float)
            Mat , status = cv2.findHomography(img_pts, pts)
            res = cv2.warpPerspective(self.img, Mat, (frame.shape[1], frame.shape[0]))
            cv2.fillConvexPoly(frame, pts.astype(int), 0, 16)
            resImg = cv2.add(frame, res)
            cv2.imshow('a' , resImg)
            if cv2.waitKey(30) == ord('q'):      # 每30毫秒更新一次，直到按下 q 結束
                    break
                

    def imageReconstruction(self):
        self.rgbImgs = []
        self.reconstructImg = []
        for img in self.imgs:
            copyImg = copy.deepcopy(img)
            cimg = cv2.cvtColor(copyImg , cv2.COLOR_BGR2RGB)
            self.rgbImgs.append(cimg)
            blue , green , red = cv2.split(cimg)
            df_blue = blue / 255
            df_green =  green/255
            df_red =  red/255
            pca_b = PCA(n_components=10)
            pca_b.fit(df_blue)
            trans_pca_b = pca_b.transform(df_blue)
            pca_g = PCA(n_components=10)
            pca_g.fit(df_green)
            trans_pca_g = pca_g.transform(df_green)
            pca_r = PCA(n_components=10)
            pca_r.fit(df_red)
            trans_pca_r = pca_r.transform(df_red)
            b_arr = pca_b.inverse_transform(trans_pca_b)
            g_arr = pca_g.inverse_transform(trans_pca_g)
            r_arr = pca_r.inverse_transform(trans_pca_r)
            img_reduced= (cv2.merge((b_arr, g_arr, r_arr)))
            self.reconstructImg.append(img_reduced)
        fig = plt.figure()
        for i , img in enumerate(self.rgbImgs , start=1):
            if(i <16):
                plt.subplot(4,15,i)
                plt.imshow(img)
                plt.axis('off')
            else:
                plt.subplot(4,15,15 + i)
                plt.imshow(img)
                plt.axis('off')     
        for i , img in enumerate(self.reconstructImg , start=1):
            if(i <16):
                plt.subplot(4,15,15 + i)
                plt.imshow(img)
                plt.axis('off')
            else:
                plt.subplot(4,15,30 + i)  
                plt.imshow(img)
                plt.axis('off')

        plt.show()

        
    def computeTheReconstructionError(self):
        rgb = []
        for img in self.rgbImgs:
            Img = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
            rgb.append(Img)
        for img in self.reconstructImg:
            Img = cv2.cvtColor(img.astype(np.float32) , cv2.COLOR_RGB2GRAY)
            rgb.append(Img)
        error = []
        total = 0
        for i in range(len(rgb)):
            total = 0
            if(i > 29):
                break
            for j in range(rgb[0].shape[0]):
                for k in range(rgb[0].shape[1]):
                    total += math.pow(rgb[i][j][k] - (rgb[i + 30][j][k] * 255) , 2)
            error.append(math.sqrt(total))
        print('reconstruction error')
        print(error)
        
            
        
    
   

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())