from PIL import Image
from Individual import Individual
import numpy as np
from random import choices
from itertools import combinations
import random
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
        self.population = np.array([Individual(width=width, height=height, gene_count=gene_count) for _ in range(population_size)])
        self.stats = [0 for _ in range(max_iter)]
        self.best_individual_per_gen = [None for _ in range(max_iter)]

        for gen in range(max_iter):
            print(f"Generation: {gen+1}")
            fitness = np.array([self.__fitness_function(ind) for ind in self.population])

            stats = (np.average(fitness), np.max(fitness), np.min(fitness))
            best_individual_index = np.argmin(fitness)
            best_individual = self.population[best_individual_index]

            # selection
            self.population = self.__selection_by_probablity(fitness)
            
            # crossover
            self.population = self.__crossover()

            # mutation
            self.population = self.__mutation()

            # find the best individual of the current gen
            self.stats[gen] = stats
            self.best_individual_per_gen[gen] = best_individual
            if gen % 10 == 0:
                print(f'Best fitness: {self.__fitness_function(best_individual)}')
                best_individual.render()

    """
        Compute the mean squere error pixel-wise between the target image
        and the current image
    """ 
    def __fitness_function(self, ind: Individual) -> float:
        if not ind.get_fitness():
            differences = np.power(self.img_data - ind.get_img_data(), 2)
            ind.set_fitness(np.mean(differences))
        return ind.get_fitness()

    """
        Compute the probability to choose each individual,
        then chooses them randomly with the probability as weights.

        Lower fitness implies better individual so the actual
        probability is 1-(f(i)/sum f(j)) for each individual i
    """
    def __selection_by_probablity(self, fitness: np.array) -> np.array:
        selection_probability = 1 - (fitness/np.sum(fitness))
        selected_population = choices(self.population, weights=selection_probability, k=self.population_size)
        return np.array(selected_population)
    
    def __tournament_selection(self, fitness: np.array) -> np.array:
        current_population_size = 0
        new_pop = np.array([None for _ in range(self.population_size)])

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
    
    """
        Given two individual, create two sons.
        each son has half the genes of each parent
    """
    def __reproduce(self, mother: Individual, father: Individual) -> tuple[Individual, Individual]:
        half_genes = self.gene_count // 2

        mother_genes = mother.get_genes()
        father_genes = father.get_genes()

        son_1_genes = []
        son_2_genes = []


        for i in range(self.gene_count):
            k = random.choice([True, False])
            if k:
                son_1_genes.append(mother_genes[i])
                son_2_genes.append(father_genes[i])
            else:
                son_2_genes.append(father_genes[i])
                son_1_genes.append(mother_genes[i])

        son_1 = Individual(width=self.width, height=self.height, gene_count=self.gene_count, make_empty=True)
        son_2 = Individual(width=self.width, height=self.height, gene_count=self.gene_count, make_empty=True)
        
        son_1.set_genes(son_1_genes)
        son_2.set_genes(son_2_genes)

        return (son_1, son_2)

    """
        Generate a list of pairs (x, y) where x != y, then chooses as many random pairs
        as individual has the population and reproduce them.
    """
    def __crossover(self) -> np.array:
        posible_parents = list(combinations(range(self.population_size), 2))
        parents = choices(posible_parents, k=self.population_size*2)
        
        current_population_size = 0 
        new_pop = np.array([None for _ in range(self.population_size*2)])

        while current_population_size != self.population_size*2:
            (mother_index, father_index) = parents[current_population_size]
            mother, father = self.population[mother_index], self.population[father_index]

            (son_1, son_2) = self.__reproduce(mother, father)

            new_pop[current_population_size] = son_1
            new_pop[current_population_size+1] = son_2

            current_population_size += 2

        new_pop = sorted(new_pop, key=(lambda ind: self.__fitness_function(ind)))
        return new_pop[:self.population_size]
    
    """
        Mutate each individual of the population by a mutation rate
    """
    def __mutation(self) -> np.array:
        new_pop = np.array([None for _ in range(self.population_size)])
        for (i, ind) in enumerate(self.population):
            ind.mutate(self.__mutation_rate)
            new_pop[i] = ind
        return new_pop