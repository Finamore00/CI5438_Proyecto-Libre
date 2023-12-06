import numpy as np
import random
from PIL import Image, ImageDraw
from IPython.display import display

"""
Clase que define un punto en un espacio cartesiano
"""
class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def to_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)


"""
Clase que define un color en el espacio RGBA. Se utiliza RGBA
en vez de RGB estándar para introducir la capacidad de especi-
ficar la opacidad de un color (artibuto "alpha")
"""
class Color:
    def __init__(self, r: int, g: int, b: int, alpha: int) -> None:
        self.red = r
        self.green = g
        self.blue = b
        self.alpha = 50

    """
    Retorna los valores RGBA del color como una tupla de enteros
    """
    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.red, self.green, self.blue, self.alpha)



"""
Clase que define un triángulo en un espacio cartesiano.
Un triángulo es definido por 3 puntos en el espacio y un
color
"""
class Triangle:
    def __init__(self, coords: list[Point], color: Color) -> None:
        self.coords = coords
        self.color = color

    """
    Retorna las coordenadas del triángulo como una tupla de coordenadas
    (Que a su vez son tuplas)
    """
    def coords_to_tuples(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        return tuple([p.to_tuple() for p in self.coords])



"""
Clase que define a un individuo dentro del algoritmo genético. 
Cada individuo consiste en una lista de genes, y cada gen corres-
ponde a un triángulo como están definidos en la clase Triangle
"""
class Individual:
    def __init__(self, 
                 width: int = 300,
                 height: int = 300,
                 gene_count: int = 300) -> None:
        random.seed()
        self.genes: list[Triangle] = []
        self.width: int = width
        self.height: int = height
        self.gene_count = gene_count
        self.__img = None
        for _ in range(gene_count):
            #Generar un triángulo con coordenadas y color aleatorios
            color_values = [random.randrange(0, 256) for _ in range(4)]
            #Se aplica un margen extra de 20 pixeles para no dejar bordes blancos
            coordinates = [Point(random.randrange(-20, (self.width)+20), random.randrange(-20, (self.height)+20)) for _ in range(3)]
            self.genes.append(Triangle(coordinates, Color(*color_values)))
    
    """
    Separamos el trazado de los triángulos en otro procedimiento para
    evitar dibujar la imagen varias veces.
    """
    def __draw_triangles(self) -> None:
        if not self.__img:
            self.__img = Image.new('RGB', (self.width, self.height), 'white')
            drawer = ImageDraw.Draw(self.__img, 'RGBA')

            for tri in self.genes:
                drawer.polygon(tri.coords_to_tuples(), fill=tri.color.to_tuple())

    """
    Función que renderiza la imagen descrita por todos los genes del individuo
    """
    def render(self) -> None:
        if not self.__img:
            self.__draw_triangles()
        display(self.__img)

    """
    Función que retorna la información de pixeles de la imagen generada por los
    triángulos del individuo como tripletas en un arreglo de numpy
    """
    def get_img_data(self) -> np.ndarray:
        if not self.__img:
            self.__draw_triangles()
        return np.array(self.__img.getdata())
    
    def get_genes(self) -> [Triangle]:
        return self.genes
    
    def get_img(self):
        return self.__img
    
    def set_genes(self, new_genes: [Triangle]):
        self.genes = new_genes
        self.__img = None
        self.__draw_triangles()

    def mutate(self, mutation_rate: float):
        for i in range(self.gene_count):
            if mutation_rate >= random.random():
                color_values = [random.randrange(0, 256) for _ in range(4)]
                coordinates = [Point(random.randrange(-20, (self.width)+20), random.randrange(-20, (self.height)+20)) for _ in range(3)]
                self.genes[i] = Triangle(coordinates, Color(*color_values))

        # restart the image to draw it again
        self.__img = None
        self.__draw_triangles()