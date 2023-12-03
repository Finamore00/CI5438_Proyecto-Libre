from PIL import Image
from Individual import Individual
import numpy as np

"""
Clase que implementa el algoritmo genético y provee las interfaces
necesarias para ingresar y extraer información hacia y desde el mismo.
"""
class Genetic_Algorithm():
    def __init__(self,
                img_route: str,
                population_size: int,
                gene_count: int,
                mutation_rate: float,
                max_iter: int = 1000) -> None:
        # Put image information into Pandas dataframe
        img = Image.open(img_route)
        self.img_data = np.array(img.getdata())
        width, height = img.size

        self.__mutation_rate = mutation_rate

        self.population = []
        #Generar población de individuos
        for _ in range(population_size):
            self.population.append(Individual(width=width, height=height, gene_count=gene_count))
        