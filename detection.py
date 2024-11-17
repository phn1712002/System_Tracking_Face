import cv2, numpy as np
from typing import Any, Dict
from geo import Point, Rectangle
   
class face_Detection_Haar:
  def __init__(self, path_config_xml:str='', config_detection:Dict={}) -> None:
    self.path_config_xml = path_config_xml
    self.config_detection = config_detection
    self.detectors = cv2.CascadeClassifier(self.path_config_xml)
    
    
  def detection(self, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = self.detectors.detectMultiScale(gray, **self.config_detection, flags=cv2.CASCADE_SCALE_IMAGE)
    
    face_ROI = []
    for (fX, fY, fW, fH) in faceRects:
      rec = Rectangle([Point(x=fX, y=fY), 
                       Point(x=fX + fW, y=fY + fH), 
                       Point(x=fX, y=fY + fH), 
                       Point(x=fX + fW, y=fY)
                       ])
      face_ROI.append(rec)
    return face_ROI
  
  def detection_one_face(self, frame):
    face = None
    face_ROI = self.detection(frame)
    if len(face_ROI) != 0:
      idx_max = np.argmax([rec.area for rec in face_ROI])
      face = face_ROI[idx_max]
    return face