from db import PlateDB
from scipy.spatial import distance as dist
from collections import OrderedDict
import re 
import cv2
import numpy
import matplotlib.pyplot as plt 
import pytesseract

PlateDB = PlateDB("Project\car_plate_project.db")

def img_open(img_path: str):
    """считывание фото"""
    carplate_img = cv2.imread(img_path)
    carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
    return carplate_img

def video_open(video_path: str):
    """Cчитывает видео"""
    video = cv2.VideoCapture(video_path)
    return video

def cascade_open(cascade_path: str):
    """Считывает каскад"""
    cascade = cv2.CascadeClassifier(cascade_path)
    return cascade
    
def img_resize(image, coef_change):
    """Изменение размеров изображения"""
    width = int(image.shape[1] * coef_change / 100)
    height = int(image.shape[0] * coef_change / 100)
    options = (width, height)
    plt.axis('off')
    resized_image = cv2.resize(image, options, interpolation = cv2.INTER_AREA)
    return resized_image

def allowed_width_height(image):
    """Проверяет чтобы картинка была не нулевого размера"""
    if (image.shape[1] * image.shape[0]) != 0:
        return True
    return False

def cascade_cropped_img(img: numpy.ndarray):
    """С помощью каскадов ищет нужную область"""
    global haar_cascade
    cascade_find = haar_cascade.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 5)
    for x, y, w, h in cascade_find:
        img_cropped = img[y+15 : y+h-10, x+15 : x+w-20]
    return img_cropped

def add_car_to_db(plate):
    """Добавляет запись в бд о машине встреченной в первый раз"""
    if not(PlateDB.plate_exists(plate)):
        PlateDB.add_personal_plate(plate)
    return

def euclidean_distance(pt1, pt2):
    """Функция для вычисления евклидового расстояния между двумя точками"""
    return dist.euclidean(pt1, pt2)

def reliability(plate: str):
    """Опредиление соответствия большему соответствию строки российскому номеру"""
    match = re.search(r'[a-zA-Z]\d{3}[a-zA-Z]{2}\d{0,}', plate) 
    return match[0] if match else 'Not found' 

def add_db(plate, img, speed):
    """Добавляет запись в базу данных о зафиксированном номере"""
    plate = reliability(plate)
    if plate != "Not found":
        add_car_to_db(plate)
        PlateDB.add_plate(plate, img, speed)
    return

def img_find_plate(img_rgb, speed):
    """Ищет на фотографии номер машины и записывает его в базу данных + скорость"""
    cropped_img = cascade_cropped_img(img_rgb)
    if not(allowed_width_height(cropped_img)):
        return
    cropped_img = img_resize(cropped_img, 150)
    cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    plate = pytesseract.image_to_string(
            cropped_img_gray,
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    add_db(plate, cropped_img, speed)

def car_speed_counter(x, y, w, h, conversion_rate = 0.0002):
    """Вычисляет приблизительное значение скорости автомобиля"""
    global centroids_dict, object_id, video
    centroid = (int(x + w / 2), int(y + h / 2))
    centroids_dict[object_id] = centroid
    object_id += 1
    if len(centroids_dict) > 1:
                previous_centroids = list(centroids_dict.values())[-2]
                current_centroids = list(centroids_dict.values())[-1]
                distance_px = euclidean_distance(previous_centroids, current_centroids)                
                distance_meters = distance_px * conversion_rate
                fps = video.get(cv2.CAP_PROP_FPS)
                speed_mps = distance_meters * fps
                return int(speed_mps * 3.6) # км/час

def car_background_subtraction(img, bg_subtractor):
    """Функция которая отлавливает движущиеся объекты и вырезает их из img и даёт их скорость"""
    frame_blurred = cv2.GaussianBlur(img, (13, 13), 0)
    fg_mask = bg_subtractor.apply(frame_blurred)

    kernel = numpy.ones((3, 3), numpy.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
            if (cv2.contourArea(contour) < 13000):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            car_speed = car_speed_counter(x, y, w, h)
            img_cropped = img[y : y+h, x : x+w]
            img_rgb_cropped = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_find_plate(img_rgb_cropped, car_speed) 

def main():
    global video, centroids_dict, object_id, haar_cascade
    video = video_open("Project\\videos\\test.mp4")
    centroids_dict = OrderedDict()
    object_id = 0
    haar_cascade = cv2.CascadeClassifier('Project\haarcascade.xml')
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    while video.isOpened():
        flag, img = video.read()
        img = cv2.resize(img, (1500, 900))
        if flag:
            car_background_subtraction(img, bg_subtractor) 
            for (x,y,w,h) in haar_cascade.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 6):
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.imshow('video', img)
                crop_img = img[y:y+h,x:x+w]
            cv2.waitKey(1)
    else:
        print("Error opening video file")

if __name__ == '__main__':
    main()