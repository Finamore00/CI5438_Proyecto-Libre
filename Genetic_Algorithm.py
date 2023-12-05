from PIL import Image
from Individual import Individual
import numpy as np
from random import choices, random
from itertools import combinations

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
        self.gene_count = gene_count
        self.tournament_size = tournament_size
        self.population = []
        self.population_size = population_size
        self.width = width
        self.height = height

        # create first generation
        self.population = [Individual(width=width, height=height, gene_count=gene_count) for _ in range(population_size)]
        self.stats = [0 for _ in range(max_iter)]
        self.best_individual_per_gen = [None for _ in range(max_iter)]

        for gen in range(max_iter):
            print(f"Generation: {gen+1}")
            fitness = np.array([self.__fitness_function(ind) for ind in self.population])

            stats = (np.average(fitness), np.max(fitness), np.min(fitness))
            best_individual_index = np.argmax(fitness)
            best_individual = self.population[best_individual_index]

            # selection
            self.population = self.__tournament_selection(fitness)

            # crossover
            self.population = self.__crossover()

            # mutation
            self.population = self.__mutation()

            # find the best individual of the current gen
            self.stats[gen] = stats
            self.best_individual_per_gen[gen] = best_individual

            

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
    

    def __reproduce(self, mother: Individual, father: Individual) -> (Individual, Individual):
        half_genes = self.gene_count // 2

        mother_genes = mother.get_genes()
        father_genes = father.get_genes()

        son_1_genes = mother_genes[0:half_genes] + father_genes[half_genes:]
        son_2_genes = father_genes[0:half_genes] + mother_genes[half_genes:]

        son_1 = Individual(width=self.width, height=self.height, gene_count=self.gene_count)
        son_2 = Individual(width=self.width, height=self.height, gene_count=self.gene_count)
        
        son_1.set_genes(son_1_genes)
        son_2.set_genes(son_2_genes)

        return (son_1, son_2)

    def __crossover(self) -> [Individual]:
        posible_parents = list(combinations(range(self.population_size), 2))
        parents = choices(posible_parents, k=self.population_size)
        
        current_population_size = 0 
        new_pop = [None for _ in range(self.population_size)]

        while current_population_size != self.population_size:
            (mother_index, father_index) = parents[current_population_size]
            mother, father = self.population[mother_index], self.population[father_index]

            (son_1, son_2) = self.__reproduce(mother, father)

            new_pop[current_population_size] = son_1
            new_pop[current_population_size+1] = son_2

            current_population_size += 2

        return new_pop
    
    def __mutation(self) -> [Individual]:
        new_pop = [None for _ in range(self.population_size)]
        for i in range(self.population_size):
            can_mutate = random()
            current_ind = self.population[i]
            if self.__mutation_rate >= can_mutate:
                current_ind.mutate()

            new_pop[i] = current_ind

        return new_pop