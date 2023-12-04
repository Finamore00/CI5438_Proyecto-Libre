from PIL import Image
from Individual import Individual
import numpy as np
from random import choices

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
                tournament_size: int,
                max_iter: int = 1000,) -> None:
        # Put image information into Pandas dataframe
        img = Image.open(img_route)
        self.img_data = np.array(img.getdata())
        width, height = img.size

        self.__mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.population = []
        self.population_size = population_size
        
        # create first generation
        self.population = [Individual(width=width, height=height, gene_count=gene_count) for _ in range(population_size)]

        for gen in range(max_iter):
            fitness = np.array([self.__fitness_function(ind) for ind in self.population])

            # selection
            self.population = self.__tournament_selection(fitness)

            # crossover

            # mutation
                        
            

    def __fitness_function(self, ind: Individual) -> float:
        value = 0
        for (current, target) in zip(ind.get_img_data(), self.img_data):
            value += np.linalg.norm(np.array(current) - np.array(target))
        return 1/value

    def __tournament_selection(self, fitness: np.array) -> [Individual]:
        current_population_size = 0
        new_pop = [None for _ in range(self.population_size)]

        while current_population_size != self.population_size:
            choosen_indexes = choices(range(self.population_size), k=self.tournament_size)
            
            best_fitness_index = choosen_indexes[0]
            best_fitness = fitness[best_fitness_index]
            for i in choosen_indexes:
                if best_fitness < fitness[best_fitness_index]:
                    best_fitness_index = i
                    best_fitness = fitness[best_fitness_index]
            new_pop[current_population_size] = self.population[best_fitness_index]
            current_population_size += 1

        return new_pop