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
        self.alpha = alpha

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
        for _ in range(gene_count):
            #Generar un triángulo con coordenadas y color aleatorios
            color_values = [random.randrange(0, 256) for _ in range(4)]
            #Se aplica un margen extra de 20 pixeles para no dejar bordes blancos
            coordinates = [Point(random.randrange(-20, self.width+20), random.randrange(-20, self.height+20)) for _ in range(3)]
            self.genes.append(Triangle(coordinates, Color(*color_values)))
    
    """
    Función que renderiza la imagen descrita por todos los genes del individuo
    """
    def render(self) -> None:
        canvas = Image.new('RGB', (self.width, self.height), 'white')
        drawer = ImageDraw.Draw(canvas, 'RGBA')

        for tri in self.genes:
            drawer.polygon(tri.coords_to_tuples(), fill=tri.color.to_tuple())

        display(canvas)
