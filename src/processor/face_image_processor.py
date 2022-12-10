from .image_handler import get_image_as_nparray_rgb
import numpy as np
from PIL import Image
from .face_image_structures import *
import logging
import ntpath
import traceback

class FaceImageProcessor:
    """Responsible to load images and process them to extract faces, embeddings, and analyze.

    Exceptions:
        FileNotFoundError: If the image file cannot be found.
        PIL.UnidentifiedImageError: If the image cannot be opened and identified.
        ValueError: When the the embedding extractor or face analysis fail because no face were detected.
    """
    image_path: str
    image_name: str
    image_nparray: np.ndarray
    image_pillow_object: Image.Image
    
    detector_type: FaceDetectorTypes
    align: bool
    
    faces: list[Face] = None
    face_image: FaceImage = None
    
    def __init__(self, image_path: str, detector_type: FaceDetectorTypes = FaceDetectorTypes.RETINAFACE, align: bool = True) -> None:
        self.image_path = image_path
        self.image_name = ntpath.basename(self.image_path)
        
        logging.debug(f"FaceImageProcessor: Reading '{self.image_name}' image.")
        try:
            image_handler_result = get_image_as_nparray_rgb(self.image_path)
        except Exception as e:
            logging.error(f"FaceImageProcessor: Failed to Read '{self.image_name}' image: {str(traceback.format_exc())}.")
            raise
        
        self.image_pillow_object = image_handler_result[0]
        self.image_nparray = image_handler_result[1]
        
        self.detector_type = detector_type
        self.align = align

    def detect_faces(self, force_detecting: bool= False) -> list[Face]:
        """Gets the bounding boxes of the faces detected in the image.
        This also save and align the detected faces.
        Note: this function does not extract the embeddings.
        
        Args:
            force_detecting (bool, optional): Force running the detecting step again. Defaults to False.
        
        Returns:
            list[Face]: List containing the faces information.
        """
        from deepface.detectors import FaceDetector
        
        if self.faces != None and not force_detecting:
            #Prevent running the detecting again.
            return self.faces
        
        deepface_detector = FaceDetector.build_model(self.detector_type.value)
        
        logging.info(f"FaceImageProcessor: Detecting faces in '{self.image_name}'.")
        detected_faces = FaceDetector.detect_faces(deepface_detector, self.detector_type.value, self.image_nparray, align=self.align)
        
        faces = [
                Face(
                    FaceBoundingBox(*face[1]),
                    face[0]
                ) for face in detected_faces
            ]
        
        logging.info(f"FaceImageProcessor: Detected {len(faces)} face/s in '{self.image_name}'.")
        self.faces = faces
        
        return self.faces

    def __extract_faces_embeddings(self, face_embedder_type: FaceEmbedderTypes):
        """Extract found faces embeddings.

        Args:
            face_embedder_type (FaceEmbedderTypes): Which model to be used to extract the embeddings.

        """
        from deepface import DeepFace
        
        logging.info(f"FaceImageProcessor: Extracting {len(self.faces)} face/s embedding in '{self.image_name}'.")
        for face in self.faces:
            try:
                embedding = DeepFace.represent(face.nparray, 
                                            model_name= face_embedder_type.value, 
                                            enforce_detection = False)
            except Exception as e:
                logging.error(f"FaceImageProcessor: Failed to extract face embeddings in '{self.image_name}': '{str(traceback.format_exc())}'")
                raise
            
            face.embedding = embedding
            
            
    def process(self, face_embedder_type: FaceEmbedderTypes, analysis_types: list[FaceAnalysisTypes] = None, force_processing: bool= False) -> FaceImage:
        """Process the image and extract face information. Like face embedding, age, gender, emotion, and race.

        Args:
            face_embedder_type (FaceEmbedderTypes): Which model to be used to extract the embeddings.
            analysis_types (list[FaceAnalysisTypes], optional): List of additional analysis to be done on the faces. Like age, gender... . Defaults to None.
            force_processing (bool, optional): Force running the processing step again. Defaults to False.

        Returns:
            FaceImage: Contains the result of the processing of the faces in the image.
        """
        from deepface import DeepFace
        
        if self.face_image != None and not force_processing:
            #Prevent running the processing again.
            return self.face_image
        
        if self.faces == None:
            #Detecting step was not done.
            self.detect_faces()
        
        self.__extract_faces_embeddings(face_embedder_type)
        
        if analysis_types != None:
            analysis_list = [analysis.value for analysis in analysis_types]
            
            logging.info(f"FaceImageProcessor: Analyzing '{self.image_name}' face/s for {', '.join(analysis_list)}.")
            
            faces_nparray = [face.nparray for face in self.faces]
            
            try:    
                anaylysis_result = DeepFace.analyze(faces_nparray, analysis_list, enforce_detection= False, prog_bar= False)
                logging.debug(f"FaceImageProcessor: Analysis result: {anaylysis_result}")
            except Exception as e:
                logging.info(f"FaceImageProcessor: Failed to analyze '{self.image_name}' face/s for {', '.join(analysis_list)}: {str(traceback.format_exc())}.")
                raise
            
        
            for index, face in enumerate(self.faces):
                for analysis in analysis_list:
                    
                    if(analysis == FaceAnalysisTypes.AGE.value):
                        #Converting age to float.
                        face.__setattr__(analysis, 
                                    float(anaylysis_result[f"instance_{index+1}"][analysis]))
                        continue
                        
                    face.__setattr__(analysis, 
                                    anaylysis_result[f"instance_{index+1}"][analysis])
                    
                    #Setting dominant attribute for Emotion and Race
                    if(analysis != FaceAnalysisTypes.AGE.value and
                       analysis != FaceAnalysisTypes.GENDER.value):
                        
                        current_analysis_result = anaylysis_result[f"instance_{index+1}"][analysis]
                        logging.debug(f"FaceImageProcessor: Setting dominant attribute between: {current_analysis_result}")
                        
                        if(analysis == FaceAnalysisTypes.EMOTION.value):
                            face.__setattr__("dominant_emotion", 
                                             FaceAnalysisEmotion(max(current_analysis_result,
                                                 key=current_analysis_result.get))
                                             )
                            
                        if(analysis == FaceAnalysisTypes.RACE.value):
                            face.__setattr__("dominant_race", 
                                             FaceAnalysisRace(max(current_analysis_result,
                                                 key=current_analysis_result.get))
                                             )
        
        self.face_image = FaceImage(self.image_name, self.image_pillow_object, self.faces)
        return self.face_image
            
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="face_embedding.log",
                        filemode="a", datefmt="%Y-%m-%d %H:%M:%S %Z",
                        format="%(asctime)s::%(levelname)s - %(message)s")
    processor = FaceImageProcessor("../test_images/DSC_0371.JPG")
    result = processor.process(FaceEmbedderTypes.FACENET512,
                      [FaceAnalysisTypes.AGE,
                       FaceAnalysisTypes.EMOTION,
                       FaceAnalysisTypes.GENDER,
                       FaceAnalysisTypes.RACE])
    
