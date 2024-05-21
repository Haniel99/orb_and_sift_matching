import cv2
import matplotlib.pyplot as plt
import numpy as np
import time  
from abc import ABC, abstractmethod

class FeatureDetector(ABC):
    """
    Clase base para detectores de características.
    """

    def __init__(self, image, width=600, height=600):
        """
        Inicializa el detector de características.
        
        Args:
            image (ndarray): Imagen en la que se detectarán las características.
            width (int): Ancho al que se redimensionará la imagen. Default es 600.
            height (int): Altura a la que se redimensionará la imagen. Default es 600.
        """
        self.resized_image = cv2.resize(image, (width, height))
        self.gray = cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2GRAY)
        self.keypoints = None
        self.descriptors = None
        self.time_taken = 0

    @abstractmethod
    def detect_and_compute(self):
        """
        Método abstracto para detectar y computar los descriptores de la imagen.
        """
        pass

    def image_keypoints(self):
        """
        Dibuja los puntos clave sobre la imagen redimensionada. 
        Se puede camabiar la imagen en donde se quiere dibujas los puntos.

        Returns:
            ndarray: Imagen con los puntos clave dibujados.
        """
        return cv2.drawKeypoints(self.resized_image, self.keypoints, None, (255, 0, 0), 2)

    def detect(self):
        """
        Devuelve los puntos clave y descriptores detectados.

        Returns:
            tuple: Tupla que contiene los puntos clave y los descriptores.
        """
        return self.keypoints, self.descriptors

    def count_keypoints(self):
        """
        Cuenta el número de puntos clave detectados.

        Returns:
            int: Número de puntos clave detectados.
        """
        return len(self.keypoints) if self.keypoints is not None else 0

    def show(self, title='', figsize=(12, 8)):
        """
        Muestra la imagen con los puntos clave.

        Args:
            title (str): Título de la imagen mostrada.
            figsize (tuple): Tamaño de la figura a mostrar.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cv2.cvtColor(self.image_keypoints(), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(title)
        plt.show()

class ORB(FeatureDetector):
    """
    Oriented FAST and rBRIEF (ORB), es un algoritmo extractor de descriptores y detector de características.
    Detecta puntos claves a través de una pirámide de entrada y extrae un descriptor para cada característica,
    devolviendo sus coordenadas, el nivel de la pirámide donde se encontró la característica,
    así como su descriptor de cadena de bits asociado.
    """

    def __init__(self, image, width=600, height=600):
        """
        Inicializa el detector ORB.
        
        Args:
            image (ndarray): Imagen en la que se detectarán las características.
            width (int): Ancho al que se redimensionará la imagen. Default es 600.
            height (int): Altura a la que se redimensionará la imagen. Default es 600.
        """
        super().__init__(image, width, height)
        self.orb = cv2.ORB_create(nfeatures=500)
        start_time = time.time()  
        self.detect_and_compute()
        end_time = time.time() 
        self.time_taken = end_time - start_time 

    def detect_and_compute(self):
        """
        Detecta y computa los puntos clave y descriptores utilizando el algoritmo ORB.
        """
        self.keypoints, self.descriptors = self.orb.detectAndCompute(self.gray, None)
    
class SIFT(FeatureDetector):
    """
    Scale-Invariant Feature Transform (SIFT), es un algoritmo extractor de descriptores y detector de características.
    Detecta puntos claves a través de una pirámide de entrada y extrae un descriptor para cada característica.
    """

    def __init__(self, image, width=600, height=600):
        """
        Inicializa el detector SIFT.
        
        Args:
            image (ndarray): Imagen en la que se detectarán las características.
            width (int): Ancho al que se redimensionará la imagen. Default es 600.
            height (int): Altura a la que se redimensionará la imagen. Default es 600.
        """
        super().__init__(image, width, height)
        self.sift = cv2.SIFT_create()
        start_time = time.time()  
        self.detect_and_compute()
        end_time = time.time() 
        self.time_taken = end_time - start_time  

    def detect_and_compute(self):
        """
        Detecta y computa los puntos clave y descriptores utilizando el algoritmo SIFT.
        """
        self.keypoints, self.descriptors = self.sift.detectAndCompute(self.gray, None)

def process_images(images, detector_class, width=600, height=600):
    """
    Procesa una lista de imágenes utilizando un detector de características específico.

    Args:
        images (list): Lista de imágenes a procesar.
        detector_class (class): Clase del detector de características a utilizar.
        width (int): Ancho al que se redimensionarán las imágenes. Default es 600.
        height (int): Altura a la que se redimensionarán las imágenes. Default es 600.

    Returns:
        list: Lista de tuplas que contienen la imagen con puntos clave dibujados, tiempo de procesamiento y número de puntos clave.
    """
    results = []
    for image in images:
        detector = detector_class(image, width, height)
        results.append((detector.image_keypoints(), detector.time_taken, detector.count_keypoints()))
    return results

def display_images_with_times(images_with_times, titles, figsize=(15, 10), algorithm_name=''):
    """
    Muestra una serie de imágenes con los tiempos de procesamiento.

    Args:
        images_with_times (list): Lista de tuplas que contienen la imagen con puntos clave dibujados, tiempo de procesamiento y número de puntos clave.
        titles (list): Lista de títulos para las imágenes.
        figsize (tuple): Tamaño de la figura a mostrar.
        algorithm_name (str): Nombre del algoritmo utilizado.
    """
    fig, axes = plt.subplots(1, len(images_with_times), figsize=figsize)
    for ax, (image, time_taken, _), title in zip(axes, images_with_times, titles):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{title}\nTime {algorithm_name}: {(time_taken*1000):.4f} ms')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def display_images_with_points(images_with_points, titles, figsize=(15, 10), algorithm_name=''):
    """
    Muestra una serie de imágenes con el número de puntos clave detectados.

    Args:
        images_with_points (list): Lista de tuplas que contienen la imagen con puntos clave dibujados, tiempo de procesamiento y número de puntos clave.
        titles (list): Lista de títulos para las imágenes.
        figsize (tuple): Tamaño de la figura a mostrar.
        algorithm_name (str): Nombre del algoritmo utilizado.
    """
    fig, axes = plt.subplots(1, len(images_with_points), figsize=figsize)
    for ax, (image, _, points), title in zip(axes, images_with_points, titles):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{title}\nKey Points {algorithm_name}: {points}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

class Matcher:
    """
    Clase para emparejar características entre dos detectores.
    """

    def __init__(self, detector1, detector2):
        """
        Inicializa el emparejador de características.

        Args:
            detector1 (FeatureDetector): Primer detector de características.
            detector2 (FeatureDetector): Segundo detector de características.
        """
        self.detector1 = detector1
        self.detector2 = detector2
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 
        self.matches = self.match_features()

    def match_features(self):
        """
        Empareja las características entre los dos detectores.

        Returns:
            list: Lista de coincidencias ordenadas por distancia.
        """
        matches = self.bf.match(self.detector1.descriptors, self.detector2.descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def get_correspondences(self):
        """
        Obtiene las correspondencias de puntos clave entre las dos imágenes.

        Returns:
            list: Lista de tuplas que contienen las coordenadas de los puntos emparejados.
        """
        correspondences = []
        for match in self.matches:
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            point1 = self.detector1.keypoints[img1_idx].pt
            point2 = self.detector2.keypoints[img2_idx].pt
            correspondences.append((point1, point2))
        return correspondences

    def draw_matches_with_numbers(self):
        """
        Dibuja los puntos emparejados con números en las dos imágenes.
        """
        img1_with_numbers = self.detector1.resized_image.copy()
        img2_with_numbers = self.detector2.resized_image.copy()

        for i, match in enumerate(self.matches):
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            point1 = self.detector1.keypoints[img1_idx].pt
            point2 = self.detector2.keypoints[img2_idx].pt

            cv2.putText(img1_with_numbers, str(i), (int(point1[0]), int(point1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img2_with_numbers, str(i), (int(point2[0]), int(point2[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if img1_with_numbers.shape != img2_with_numbers.shape:
            height = max(img1_with_numbers.shape[0], img2_with_numbers.shape[0])
            img1_with_numbers = cv2.copyMakeBorder(img1_with_numbers, 0, height - img1_with_numbers.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            img2_with_numbers = cv2.copyMakeBorder(img2_with_numbers, 0, height - img2_with_numbers.shape[0], 0, 0, cv2.BORDER_CONSTANT)

        img_matches_with_numbers = cv2.hconcat([img1_with_numbers, img2_with_numbers])
        plt.imshow(img_matches_with_numbers)
        plt.show()

    def draw_matches(self, title):
        """
        Dibuja las coincidencias entre los dos detectores.

        Args:
            title (str): Título de la imagen mostrada.
        """
        img_matches = cv2.drawMatches(self.detector1.resized_image, self.detector1.keypoints,
                                      self.detector2.resized_image, self.detector2.keypoints,
                                      self.matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches)
        plt.title(f'Matching {title}')
        plt.show()

class Main:
    """
    Clase principal para ejecutar las detecciones.
    """

    def detect_points_orb(self, path_img):
        """
        Detecta y muestra puntos clave utilizando ORB en una imagen.

        Args:
            path_img (str): Ruta de la imagen.
        """
        img1 = cv2.imread(path_img)  
        orb_detector1 = ORB(img1)
        orb_detector1.show('Key points ORB')

    def detect_points_sift(self, path_img):
        """
        Detecta y muestra puntos clave utilizando SIFT en una imagen.

        Args:
            path_img (str): Ruta de la imagen.
        """
        img1 = cv2.imread(path_img)  
        sift_detector1 = SIFT(img1)
        sift_detector1.show('Key points SIFT')

    def count_points(self):
        """
        Cuenta y muestra el número de puntos clave detectados en una serie de imágenes utilizando ORB y SIFT.
        """
        path_images = ['images/cars/car01.png', 'images/cars/imagen1.png', 'images/cars/imagen2.png', 'images/cars/imagen3.png', 'images/cars/imagen4.png']
        images = [cv2.imread(path) for path in path_images]
        orb_results = process_images(images, ORB)
        sift_results = process_images(images, SIFT)
        titles = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5']
        display_images_with_points(orb_results, titles, algorithm_name='ORB')
        display_images_with_points(sift_results, titles, algorithm_name='SIFT')
        
    def perform(self):
        """
        Detecta y muestra los puntos clave y tiempos de procesamiento en una serie de imágenes utilizando ORB y SIFT.
        """
        path_images = ['images/cars/car01.png', 'images/cars/imagen1.png', 'images/cars/imagen2.png', 'images/cars/imagen3.png', 'images/cars/imagen4.png']
        images = [cv2.imread(path) for path in path_images]
        orb_results = process_images(images, ORB)
        sift_results = process_images(images, SIFT)
        titles = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5']
        display_images_with_times(orb_results, titles, name_algorithm='ORB')
        display_images_with_times(sift_results, titles, name_algorithm='SIFT')
        
    def matching(self):
        """
        Realiza el emparejamiento de características entre dos imágenes utilizando ORB y SIFT, y muestra las coincidencias.
        """
        img1 = cv2.imread('images/patents/patent01.png')  
        img2 = cv2.imread('images/patents/patent02.png')  

        # Utilizando ORB
        orb_detector1 = ORB(img1)
        orb_detector2 = ORB(img2, width=500, height=200)
        print(f'Tiempo de procesamiento ORB detector1: {orb_detector1.time_taken:.4f} segundos')
        print(f'Tiempo de procesamiento ORB detector2: {orb_detector2.time_taken:.4f} segundos')

        # Match ORB features
        matcher_orb = Matcher(orb_detector1, orb_detector2)
        print(f'ORB correspondences: {len(matcher_orb.get_correspondences())}')
        matcher_orb.draw_matches('ORB')

        # Utilizando SIFT
        sift_detector1 = SIFT(img1)
        sift_detector2 = SIFT(img2, width=500, height=200)
        print(f'Tiempo de procesamiento SIFT detector1: {sift_detector1.time_taken:.4f} segundos')
        print(f'Tiempo de procesamiento SIFT detector2: {sift_detector2.time_taken:.4f} segundos')

        # Match SIFT features
        matcher_sift = Matcher(sift_detector1, sift_detector2)
        print(f'SIFT correspondences: {len(matcher_sift.get_correspondences())}')
        matcher_sift.draw_matches('SIFT')

if __name__ == '__main__':
    
    """
    Llamar a los metodos de la clase Main para ver los ejemplos
    """
    main_class = Main()
    path_image = 'images/cars/imagen_brillo_ajustado.png'
   
    #main_class.detect_points_orb(path_image)
    #main_class.detect_points_sift(path_image)
    #main_class.perform()
    #main_class.count_points()
    #main_class.matching()
