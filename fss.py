#FSS
from __future__ import division

import numpy as np
import pandas as pd

from abc import ABCMeta
from abc import abstractmethod
from six import add_metaclass

from copy import deepcopy
from copy import copy
from functools import reduce
from functools import partial

from matplotlib.style import use
import matplotlib.pyplot as plt

"""# Fish School Search - FSS

O QUE É FSS?
O FSS é uma família de algoritmos adequados para otimização em espaços de pesquisa de alta dimensão. Todos os peixes realizam buscas locais e a escola agrega informações sociais.

Referência: https://fbln.me/fss/about-fss/

#CONCEITOS

**Aquário**
Região no espaço de busca onde o peixe pode ser posicionado e permitido a movimentação.


**Densidade alimentar do aquário**
Relacionado à função a ser otimizada no espaço de busca


O algoritmo começa com os peixes sendo iniciados em posições aleatórias.

Os **Tipos de Operadores** podem ser agrupado em duas classes: **Feeding e Swimming**


**Feeding:** A qualidade da solução para o problema


**Swimming:** Conduz os movimentos dos peixes.


As **orientações de busca** no FSS são orientadas apenas por **peixes de sucesso**.


Para que isso aconteça, o FSS exige que os **operadores de Swimming** se conectem ao **operador de Feeding.**

#OPERADORES DO FSS

* Operador de Feeding
* Operador de Movimento Individual
* Operador de Movimento de Instinto Coletivo
* Operador de Movimento Volitivo

#STEPS DO FSS
![alt text](https://docs.google.com/uc?id=1MaQXI8mg3afvtCcsYxvkKb3clwsjrfGF)

#PSEUDO-CODIGO FSS


```
initialize randomly all fish;
while stop criterion is not met do
  for each fish do
    individual movement
      + evaluate fitness function;
      feeding operator;
  end
  for each fish do
    instinctive movement;
  end
  Calculate barycentre;
  for each fish do
    volitive movement;
    evaluate fitness function;
    end
  update stepind
end
```

#PEIXE (INDIVÍDUO)
**Indivíduo da população.**

Cada peixe possui uma determinada posição no espaço de busca do problema e esta posição representa uma solução em potencial para o problema tratado.
"""

class Fish(object):
    #Atributos de inicializacao do Peixe
    INITIAL_WEIGHT = 1.0
    INITIAL_FITNESS = np.inf
    INITIAL_FITNESS_DIFFERENCE = 0.0

    def __init__(self, obj_function):
        self.pos = obj_function.sample()
        self.weight = Fish.INITIAL_WEIGHT
        self.fitness = Fish.INITIAL_FITNESS
        self.delta_pos = np.zeros(shape=len(self.pos))
        self.delta_fitness = Fish.INITIAL_FITNESS_DIFFERENCE

"""#CARDUME"""

class FSS(object):

    #Atributos de inicializacao do Cardume
    def __init__(self, obj_function, school_size=30, n_iter=5000,
                 initial_step_in=1, final_step_in=0.01,
                 initial_step_vol=1, final_step_vol=0.01):
        self.school = []
        self.school_size = school_size
        self.obj_function = obj_function

        self.n_iter = n_iter
        self.initial_step_in = initial_step_in
        self.final_step_in = final_step_in
        self.step_in = initial_step_in

        self.initial_step_vol = initial_step_vol
        self.final_step_vol = final_step_vol
        self.step_vol = initial_step_vol
        self.progress = dict()

    #Reiniciar o processo de busca do zero (opcional)
    def __reset_optimizer(self):
        self.school = []
        self.step_in = self.initial_step_in
        self.step_vol = self.initial_step_vol
        self.optimality_tracking = []
        self.optimal_solution = None

    #Cria o cardume
    def __init_school(self):
        for _ in range(self.school_size):
            self.school.append(Fish(self.obj_function))
        self.total_weight = sum(list(map(lambda fish: fish.weight, self.school)))

    #Avaliacao do cardume
    def __evaluate_school(self):
        evaluate = partial(FSS.__evaluate_fish, obj_function=self.obj_function)
        self.school = list(map(evaluate, self.school))

    #Atualizacao da solucao otima
    def __update_optimal_solution(self):
        n_optimal_solution = min(self.school, key=lambda fish: fish.fitness)
        if not self.optimal_solution:
            self.optimal_solution = deepcopy(n_optimal_solution)
        elif n_optimal_solution.fitness <= self.optimal_solution.fitness:
                self.optimal_solution = deepcopy(n_optimal_solution)

    #Atualiza a lista dos peixes com as melhores solucoes (opcional)
    def __update_optimality_tracking(self):
        self.optimality_tracking.append(self.optimal_solution.fitness)

    #Atualiza o peso do cardume
    def __update_total_weight(self):
        self.total_weight = sum(list(map(lambda fish: fish.weight, self.school)))

    #Aplica o operador de movimento individual (Swimming)
    def __apply_individual_movement_operator(self):
        individual_movement_operator = partial(FSS.__individual_movement_operator,
                                               obj_function=self.obj_function,
                                               step=self.step_in)
        self.school = list(map(individual_movement_operator, self.school))

    #Aplica o operador de alimentacao(Feeding) do peixe
    def __apply_feeding_operator(self):
        max_delta_fitness = max(self.school, key=lambda fish: fish.delta_fitness).delta_fitness
        feeding_operator = partial(FSS.__feeding_operator, max_delta_fitness=max_delta_fitness)
        self.school = list(map(feeding_operator, self.school))

    #Aplica o operador de movimento institivo (Swimming)
    def __apply_collective_instinctive_movement_operator(self):
        sum_delta_fitness = sum(list(map(lambda fish: fish.delta_fitness, self.school)))
        instinctive_movement = list(map(lambda fish: fish.delta_fitness * fish.delta_pos, self.school))
        instinctive_movement = reduce(lambda x, y: x + y, instinctive_movement)/sum_delta_fitness
        collective_instinctive = partial(FSS.__collective_instinctive_movement_operator,
                                         instinctive_movement=instinctive_movement)
        self.school = list(map(collective_instinctive, self.school))

    #Calcula o baricentro (Memoria do Cardume)
    def __calculate_barycenter(self):
        sum_weights = sum(list(map(lambda fish: fish.weight, self.school)))
        barycenter = list(map(lambda fish: fish.pos * fish.weight, self.school))
        self.barycenter = reduce(lambda x, y: x + y, barycenter) / sum_weights
        
    #Aplica o operador de movimento volitivo (Swimming)
    def __apply_collective_volitive_movement(self):
        collective_volitive_movement = partial(FSS._collective_volitive_movement,
                                               barycenter=self.barycenter, total_weight=self.total_weight,
                                               step_vol=self.step_vol, fish_school=self.school)
        self.school = list(map(collective_volitive_movement, self.school))

    # Atualiza as iteracoes (Decaimento linear)
    def __update_step(self, itr):
        if self.step_in > self.final_step_in:
            self.step_in -= (self.final_step_in - self.initial_step_in)/(itr + 1)

        if self.step_vol > self.final_step_vol:
            self.step_vol -= (self.final_step_vol - self.initial_step_vol) / (itr + 1)

    def __update_history(self, itr):
      x = list(map(lambda f: f.pos[0], self.school))
      y = list(map(lambda f: f.pos[1], self.school))
      self.progress[itr] = (x, y)

    #Algorimto do FSS de acordo com o Pseudocodigo
    def optimize(self):
        self.__reset_optimizer()
        self.__init_school()
        for itr in range(self.n_iter):
            self.__evaluate_school()
            self.__update_optimal_solution()

            self.__apply_individual_movement_operator()
            self.__apply_feeding_operator()

            self.__evaluate_school()
            self.__update_optimal_solution()
            self.__update_optimality_tracking()

            self.__apply_collective_instinctive_movement_operator()
            self.__calculate_barycenter()
            self.__apply_collective_volitive_movement()

            self.__update_total_weight()
            self.__update_step(itr)
            self.__update_history(itr)
            print("iter: {} = cost: {}".format(itr, "%04.03e" % self.optimal_solution.fitness))

    #Avaliacao da posicao do Peixe
    @staticmethod
    def __evaluate_fish(fish, obj_function):
        fish.fitness = obj_function.evaluate(fish.pos)
        return fish

    #Delimitacao das restricoes do espaco busca
    @staticmethod
    def __evaluate_boundaries(pos, obj_function):
        if (pos < obj_function.minf).any() or (pos > obj_function.maxf).any():
            pos[pos > obj_function.maxf] = obj_function.maxf
            pos[pos < obj_function.minf] = obj_function.minf
        return pos

    #Movimento individual 
    @staticmethod
    def __individual_movement_operator(fish, obj_function, step):
        n_candidate_pos = np.random.uniform(low=-1, high=1, size=obj_function.dim) * step + fish.pos
        n_candidate_pos = FSS.__evaluate_boundaries(n_candidate_pos, obj_function)
        n_fitness_evaluation = obj_function.evaluate(n_candidate_pos)
        fish.delta_fitness = n_fitness_evaluation - fish.fitness
        if n_fitness_evaluation < fish.fitness:
            fish.pos = n_candidate_pos
            fish.fitness = n_fitness_evaluation
        return fish

    #Alimentacao do peixe
    @staticmethod
    def __feeding_operator(fish, max_delta_fitness):
        fish.weight += fish.delta_fitness/max_delta_fitness
        return fish

    #Movimento Institivo
    @staticmethod
    def __collective_instinctive_movement_operator(fish, instinctive_movement):
        fish.pos += instinctive_movement
        return fish

    #Movimento Volitivo
    @staticmethod
    def _collective_volitive_movement(fish, barycenter, total_weight, step_vol, fish_school):
        current_total_weight = sum(list(map(lambda fish: fish.weight, fish_school)))
        off_set = step_vol * np.random.uniform(low=0.1, high=0.9) *\
                  (fish.pos - barycenter) / np.linalg.norm(fish.pos - barycenter)
        if current_total_weight > total_weight:
            fish.pos -= off_set
        else:
          fish.pos += off_set
        return fish

"""# INICIALIZAÇÃO DO PARÂMETROS DO FSS

**Tamanho da população (Tamanho do Cardume):** pop (1 ≤ i ≤ pop)

**Dimensão do Problema:** dim (1 ≤ j ≤ dim)

**Peso do Peixe i:** wi

**Posição do Peixe i:** ~xi

**Fitness do Peixe i:** f(~xi)

# DEFINIÇÃO BASE DA FUNÇÃO OBJETIVO
"""

@add_metaclass(ABCMeta)
class ObjectiveFunction(object):

    def __init__(self, name, dim, minf, maxf):
        self.name = name
        self.dim = dim
        self.minf = minf
        self.maxf = maxf

    def sample(self):
        return np.random.uniform(low=self.minf, high=self.maxf, size=self.dim)

    def custom_sample(self):
        return np.repeat(self.minf, repeats=self.dim) \
               + np.random.uniform(low=0, high=1, size=self.dim) *\
               np.repeat(self.maxf - self.minf, repeats=self.dim)

    @abstractmethod
    def evaluate(self, x):
        pass

class Sphere(ObjectiveFunction):

    def __init__(self, dim):
        super(Sphere, self).__init__('Sphere', dim, -100.0, 100.0)

    def evaluate(self, x):
        return sum(np.power(x, 2))

objective_function = Sphere(dim=2)

"""#INICIALIZAÇÃO DO CARDUME

"""

school = [] 
school_size = 30
for _ in range(school_size):
  school.append(Fish(objective_function))
  total_weight = sum(map(lambda fish: fish.weight, school))

"""
## EXIBIR O CARDUME"""

for f in school:
    # posição do cardume
    x, y = zip(f.pos)
    plt.plot(x,y, marker='o')

plt.plot(0,0, marker='*') # Origem do Plano Cartesiano
plt.axis([-100, 100, -100, 100])
plt.show()

"""## VISUALIZAÇÃO DA FUNÇÃO DE FITNESS

### Sphere Function

$$F_{Sphere} = \sum_{i=1}^{D} x_i^2$$

![alt text](http://benchmarkfcns.xyz/benchmarkfcns/plots/spherefcn.png)

#SIMULAÇÃO E RESULTADOS
"""

use('classic')

def simulate(obj_function, school_size=30, n_iter=10000, initial_step_in=0.1, final_step_in=0.001,
             initial_step_vol=0.01, final_step_vol=0.001, simulations=1):
    itr = range(n_iter)
    values = np.zeros(n_iter)
    for _ in range(simulations):
        optimizer = FSS(obj_function, school_size=school_size, n_iter=n_iter,
                        initial_step_in=initial_step_in, final_step_in=final_step_in,
                        initial_step_vol=initial_step_vol, final_step_vol=final_step_vol)
        optimizer.optimize()
        values += np.array(optimizer.optimality_tracking)
    values /= simulations

    title = 'FSS - ' + 'Sphere' + ' Function'
    plt.plot(itr, values, lw=0.5, label=obj_function)
    plt.legend(loc='upper right')
    plt.title(title)
    return optimizer


objective_function = Sphere(2)
optimizer = simulate(obj_function=objective_function, n_iter=100)
plt.show()

fig, axis = plt.subplots(nrows=2, ncols=5, figsize=(18, 8))
axis = axis.ravel()
progress = optimizer.progress

iterations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
for idx, (ax, itr) in enumerate(zip(axis, iterations)):
    x, y = progress[itr]
    ax.scatter(x,y, marker='o')
    ax.scatter(0, 0, marker='*')
    ax.axis([-100, 100, -100, 100])
    ax.set_title('Iteracao N = {}'.format(idx))

fig.tight_layout()