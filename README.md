# CI5438 - Proyecto Libre - Sep-Dic 2023
Autores: Santiago Finamore (18-10125) & Daniel Robayo (18-11086)

## Para ejecutar
Para ejecutar el proyecto solo es necesario contar con la capacidad de ejecutar cuadernos *Jupyter Notebook*. Adicionalmente, el proyecto hace uso de las siguientes librerías:

* Python Image Library (PIL)
* Pandas
* Numpy
* colour-science (Nota: Puede crear conflictos si se hace uso de la librería "colour")
* IPython

Puede instalar estas librerías usando pip o el distribuidor de paquetes de su preferencia.

## Parámetros de la clase Genetic_Algorithm

La clase Genetic_Algorithm cuenta con un número de parámetros ajustables que condicionan el comportamiento del algoritmo.

* img_route: String que contiene la ruta dentro del sistema de archivos de la imagen a replicar
* population_size: Determina el número de individuos total de la población, este número no varía entre generaciones.
* gene_count: Indica el número de genes por individuo. Todos los individuos tienen el mismo número de genes y este no varía entre generaciones.
* mutation_rate: Valor float entre 0 y 1 que indica al programa la probabilidad de efectuar una mutación en un gen.
* use_deltaE: Valor booleano que indica al programa la función de fitness a utilizar. Si es True utiliza deltaE, si es false utiliza MSE. Este valor es False por defecto.
* max_iter: Número máximo de generaciones a generar. Por defecto 1000.

El algoritmo se ejecuta al momento de instanciar la clase, por lo que no es necesario efectuar invocación alguna de métodos posteriores.
