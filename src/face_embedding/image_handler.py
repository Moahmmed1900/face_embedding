from PIL import Image
import numpy as np

def get_image_as_nparray_rgb(image_path: str, convert_to: str = "RGB") -> tuple[Image.Image, np.ndarray]:
    """Load the specified image in Numpy array.

    Args:
        image_path (str): Path to image.
        convert_to (str, optional): Which type to load the image as. Defaults to "RGB".

    Returns:
        np.ndarray: Contains the RGB converted image as numpy array.
        Image.Image: The loaded image as Pillow Image.
    
    Exceptions:
        FileNotFoundError: If the file cannot be found.
        PIL.UnidentifiedImageError: If the image cannot be opened and identified.
    """

    image_file = Image.open(image_path)
    
    image = image_file.convert(convert_to)
    
    nparray = np.asarray(image)
    
    return (image_file, nparray)
