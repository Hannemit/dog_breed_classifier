import cv2


# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('C:\ProgramData\Anaconda3\envs\dog_breed\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')


def face_detector(img_path: str):
    """
    function that returns True if a human face is detected in the image, False otherwise
    :param img_path: string, path to an image
    :return: boolean, True if human detected, False if not. If no image is found at the specified path, return False
    """
    img = cv2.imread(img_path)

    # if no image at that path, return False
    if img is None:
        return False

    # convert to grey
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces. If no face detected, it's empty and len(faces) will be 0
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
