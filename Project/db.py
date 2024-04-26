import numpy as np
import sqlite3
import cv2

class PlateDB:
    def __init__(self, db_file):
        """Инициализация соединения с бд"""
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()

    def plate_exists(self, personal_plate):
        """Проверяем наличие машины в бд"""
        result = self.cursor.execute("SELECT `car_id` FROM `cars` WHERE `personal_plate` = ?", (personal_plate,))
        return bool(len(result.fetchall()))

    def get_personal_plate(self, personal_plate):
        """Достаем car_id машины в бд по его personal_plate"""
        result = self.cursor.execute("SELECT `car_id` FROM `cars` WHERE `personal_plate` = ?", (personal_plate,))
        return result.fetchone()[0]

    def add_personal_plate(self, personal_plate):
        """Добавляем уникальный номер машины в бд"""
        self.cursor.execute("INSERT INTO `cars` (`personal_plate`) VALUES (?)", (personal_plate,))
        return self.conn.commit()
    
    def add_plate(self, plate, img, speed):
        """Добавляем запись о зафиксированном номере машины в бд"""
        car_id = self.get_personal_plate(plate)
        img = self.convert_to_byte(img)
        self.cursor.execute("INSERT INTO `records` (`car_id`, `plate`, `photo`, `speed`) VALUES (?, ?, ?, ?)",
            (car_id,
            plate,
            img,
            speed))
        return self.conn.commit()
    
    def convert_to_byte(self, img):
        """Конвертирует <class 'numpy.ndarray'> в <class 'bytes'>"""
        img_encode = cv2.imencode('.jpg', img)[1]
        data_encode = np.array(img_encode)  
        byte_encode = data_encode.tobytes() 
        return data_encode.tobytes()
    
    def get_photo_plate(self, plate, count = 1, car_id = None):
        """Достаёт из базы данных фото номера по распознанному тексту"""
        count -= 1
        if car_id is None:
            car_id = self.get_personal_plate(plate)
            result = self.cursor.execute("SELECT `photo` FROM `records` WHERE `car_id` = ?", (car_id,))
        else:
            result = self.cursor.execute("SELECT `photo` FROM `records` WHERE `plate` = ?", (plate,))
        return result.fetchone()[count]

    def write_to_file(self, writeto_path, plate, car_id = None, count = 1):
        """Записывает фото в файл"""
        if car_id is None:
            data = self.get_photo_plate(plate, count)
        else:
            data = self.get_photo_plate(plate, count, car_id)
        with open(writeto_path, 'wb') as file:
            file.write(data)
        print("Данный из blob сохранены в: ", writeto_path, "\n")
        return
    
    def close(self):
        """Закрываем соединение с бд"""
        self.connection.close()
