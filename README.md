## About
Simple Wrapper to detect all faces in an image and extract their embeddings.

## Installation
```sh
git clone https://github.com/Moahmmed1900/face_embedding.git
cd face_embedding
pip3 install dist\face_embedding-0.0.1.tar.gz"
```

## Usage

```python
from face_embedding.face_image_processor import FaceImageProcessor
from face_embedding.face_image_structures import *

processor:FaceImageProcessor = FaceImageProcessor("./image_name.JPG")
result:FaceImage = processor.process(FaceEmbedderTypes.FACENET512,
                    [FaceAnalysisTypes.AGE,
                    FaceAnalysisTypes.EMOTION,
                    FaceAnalysisTypes.GENDER,
                    FaceAnalysisTypes.RACE])
```

## Dependencies

| Dependency | Version | Link |
| ------ | ------ | ------ |
| DeepFace | 0.0.75 | https://github.com/serengil/deepface
| Tensorflow | 2.9.0 | https://tensorflow.org/install