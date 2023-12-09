from PIL import Image
from Individual import Individual
import numpy as np
from random import choices
import random
from colour.difference import delta_E_CIE1976
import matplotlib.pyplot as plt

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
                use_deltaE: bool = False,
                max_iter: int = 1000,) -> None:
        # Put image information into Pandas dataframe
        img = Image.open(img_route)
        self.img_data = np.array(img.getdata())
        width, height = img.size

        self.__mutation_rate = mutation_rate
        self.gene_count = gene_count
        self.population = []
        self.population_size = population_size
        self.width = width
        self.height = height
        self.use_deltaE = use_deltaE
        self.num_generation = max_iter

        # create first generation
        self.population = np.array([Individual(width=width, height=height, gene_count=gene_count) for _ in range(population_size)])
        self.stats = [0 for _ in range(max_iter)]

        for gen in range(max_iter):
            fitness = np.array([self.__fitness_function(ind) for ind in self.population])

            best_individual_index = np.argmin(fitness)
            best_individual = self.population[best_individual_index]

            # selection
            self.population = self.__selection(fitness)
            
            # mutation
            self.population = self.__mutation()

            # find the best individual of the current gen
            self.stats[gen] = np.round(np.average(fitness))
            if gen % 100 == 0:
                print(f'Gen {gen} -> Best fitness: {self.__fitness_function(best_individual)}')
                best_individual.render()
        self.plot_avg_fitness()

    def plot_avg_fitness(self):
        x = np.linspace(1, self.num_generation, self.num_generation)
        y = self.stats

        ax = plt.subplot()
        ax.plot(x, y)
        ax.set_title("Funcion fitness con respecto a las generaciones.")
        ax.set_xlabel("Generacion")
        ax.set_ylabel("Fitness")
        plt.show()

    """
        Compute the mean squere error pixel-wise between the target image
        and the current image
    """ 
    def __fitness_function(self, ind: Individual) -> float:
        if not ind.get_fitness():
            if self.use_deltaE:
                differences = delta_E_CIE1976(self.img_data, ind.get_img_data()) 
            else:
                differences = np.power(self.img_data - ind.get_img_data(), 2)
            ind.set_fitness(np.mean(differences))
        return ind.get_fitness()


    def __selection(self, fitness: np.array) -> np.array:
        selection_probability = 1 - (fitness/np.sum(fitness))
        new_population = [None for _ in range(self.population_size*2)]
        new_population_size = 0

        while new_population_size != self.population_size*2:
            son = self.__crossover(selection_probability)
            new_population[new_population_size] = son
            new_population_size += 1

        sorted_population = sorted(new_population, key=(lambda ind: self.__fitness_function(ind)))
        return sorted_population[: self.population_size]
     
    """
        Choose two parents by probabilty and recombine thei genes
        in two sons. Then, return the best
    """
    def __crossover(self, selection_probability) -> Individual:
        (mother, father) = choices(self.population, weights=selection_probability, k=2)
        half_genes = self.population_size // 2

        mother_genes = mother.get_genes()
        father_genes = father.get_genes()

        """
        son_1_genes = mother_genes[:half_genes] + father_genes[half_genes:]
        son_2_genes = father_genes[:half_genes] + mother_genes[half_genes:]
        """

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

        # return the best son
        if self.__fitness_function(son_1) > self.__fitness_function(son_2):
            return son_2
        return son_1
    
    """
        Mutate each individual of the population by a mutation rate
    """
    def __mutation(self) -> np.array:
        new_pop = np.array([None for _ in range(self.population_size)])
        for (i, ind) in enumerate(self.population):
            ind.mutate(self.__mutation_rate)
            new_pop[i] = ind
        return new_pop