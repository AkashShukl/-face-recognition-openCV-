import cv2,os
import numpy as np
from sklearn import svm,metrics
from sklearn.externals import joblib

svm_model1= joblib.load('hoglinear.pkl')
svm_model2= joblib.load('hogpoly.pkl')
svm_model3= joblib.load('hogrbf.pkl')

winSize=(64,64)
blockSize=(16,16)
blockStride=(8,8)
cellSize=(8,8)
nbins=9
derivAperture=1
winSigma=4
histogramNormType=0
L2HysThreshold=2.00000000000000001e-01
gammaCorrection=0
nlevels=64
hog=cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,

                      L2HysThreshold,gammaCorrection,nlevels)

names=["0",'Akash','vaseem_sir','random1','poonam_mam','random3',"monika Jha"]
cam=cv2.VideoCapture(0)

detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):

            ret,img=cam.read()
            img1=img
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=detector.detectMultiScale(gray,1.1,7)


            for(x,y,w,h) in faces:
                img=gray[y:y+h,x:x+w]
                img=cv2.resize(img,(200,200))
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                hist=hog.compute(img,(16,16) ,(16,16),((10,20),))
                data=np.reshape(hist,(1,-1))
                op1,op2,op3=svm_model1.predict(data),svm_model2.predict(data),svm_model3.predict(data)
                print(op1,op2,op3)
                #print(metrics.accuracy_score(test_target,op)) 
                
                '''#putting Text
                font= cv2.FONT_HERSHEY_SIMPLEX
                location = x,y
                fontScale= 1
                fontColor= (255,255,255)
                lineType = 2
                print(op)
                cv2.putText(img1,'Hello '+names[int(op)], location,font,fontScale,fontColor,lineType)
               ''' 
            cv2.imshow('frame',img1)
            if cv2.waitKey(1) &0xFF==ord('q'):
                break
cam.release()
cv2.destroyAllWindows()

