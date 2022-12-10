from dataclasses import dataclass, field
from enum import Enum
import logging
from PIL import Image
import numpy as np

class FaceDetectorTypes(str, Enum):
    OPENCV = "opencv"
    SSD = "ssd"
    DLIB = "dlib"
    MTCNN = "mtcnn"
    RETINAFACE = "retinaface"
    
    
class FaceEmbedderTypes(str, Enum):
    VGG = "VGG-Face"
    FACENET = "facenet"
    FACENET512 = "Facenet512"
    OPENFACE = "OpenFace"
    DEEPFACE = "DeepFace"
    DEEPID = "DeepID"
    DLIB = "Dlib"
    ARCFACE = "ArcFace"
    
class FaceAnalysisTypes(str, Enum):
    AGE = "age"
    GENDER = "gender"
    EMOTION = "emotion"
    RACE = "race"
    
class FaceAnalysisGender(str, Enum):
    WOMEN = "Woman"
    MAN = "Man"
    
class FaceAnalysisEmotion(str, Enum):
    SAD = "sad"
    ANGRY = "angry"
    SURPRISE = "surprise"
    FEAR = "fear"
    HAPPY = "happy"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    
class FaceAnalysisRace(str, Enum):
    INDIAN = "indian"
    ASIAN = "asian"
    LATINOHISPANIC = "latino hispanic"
    BLACK = "black"
    MIDDLEEASTERN = "middle eastern"
    WHITE = "white"

@dataclass
class FaceBoundingBox():
    x: int
    y: int
    width: int
    height: int
    
    def toJson(self):
        logging.debug("FaceBoundingBox: Converting FaceBoundingBox object to json.")
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height)
        }
        
    def __repr__(self) -> str:
        return str(self.to_json())
    
    
@dataclass
class Face:
    bounding_box: FaceBoundingBox
    nparray: np.ndarray
    
    embedding: list = field(default= None, init=False)
    
    age: float = field(default= None, init=False)
    
    gender: str = field(default= None, init=False)
    
    emotion: dict = field(default= None, init=False)
    dominant_emotion: FaceAnalysisEmotion = field(default= None, init= False)
    
    race: dict = field(default= None, init=False)
    dominant_race: FaceAnalysisRace = field(default= None, init= False)
    
    def get_face_image(self):
        return Image.fromarray(self.nparray)
    
    def to_json(self):
        logging.debug("Face: Converting Face object to json.")
        return {
            "bounding_box": self.bounding_box.toJson(),
            "embeddings" : self.embedding,
            
            "age": self.age,
            
            "gender": self.gender,
            
            "emotion": self.emotion,
            "dominant_emotion": self.dominant_emotion,
            
            "race": self.race,
            "dominant_race": self.dominant_race
        }
        
    def __repr__(self) -> str:
        return str(self.to_json())
    
@dataclass
class FaceImage:
    image_name: str
    original_image: Image.Image
    faces: list[Face]
    
    def to_json(self):
        logging.debug("FaceImage: Converting FaceImage object to json.")
        return {
            "image_name": self.image_name,
            "faces": [face.to_json() for face in self.faces]
        }
    
    def __repr__(self) -> str:
        return str(self.to_json())
