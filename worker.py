import numpy as np
from random import randrange, randint, random, seed, sample
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()
LENGTH = 29
POPULATION_SIZE = 300                    # кол-во индивидуумов в популяции
AMOUNT_CHILDS = POPULATION_SIZE // 10    # кол-во детей переходящих в следующее поколение
P_CROSSOVER = 0.5                        # вероятность скрещивания
P_MUTATION = 0.001                       # вероятность мутации
NUM_GENERATIONS = 150                    # количество поколений

RANDOM_SEED = 10000                      # начальное значение для генератора псевдослучайных чисел
seed(RANDOM_SEED)
    
x = [1150, 630, 40, 750, 750, 1030, 1650, 1490, 790, 710, 840, 1170, 970, 510, 750, 1280, 230, 460, 1040, 590, 830, 490, 1840, 1260, 1280, 490, 1460, 1260, 360]
y = [1760, 1660, 2090, 1100, 2030, 2070, 650, 1630, 2260, 1310, 550, 2300, 1340, 700, 900, 1200, 590, 860, 950, 1390, 1770, 500, 1240, 1500, 790, 2130, 1420, 1910, 1980]
    
distance_matrix = []
for i in range(len(x)):
    line = []
    for j in range(len(y)):
        line.append(round(((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2) ** 0.5))
    distance_matrix.append(line)
    
class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness_value = 0
    
def fitness_func(nums):
    count = distance_matrix[nums[-1]][nums[0]]
    for i in range(LENGTH - 1):
        x = nums[i]
        y = nums[i + 1]
        count += distance_matrix[x][y]
    return count
    
# ф-ция создания индивида
def individual_creator():
    return Individual(sample(range(LENGTH), LENGTH))
    
# ф-ция создания популяции
def population_creator(n):
    return [individual_creator() for i in range(n)]
    
# ф-ция вычисления значения fitness функции
def calculations(population):
    for individ in population:
        individ.fitness_value = fitness_func(individ)
    
# функция турнирного отбора
def selection_tournament(population, p_len=POPULATION_SIZE):
    temp_population = []
    for n in range(p_len):
        temp = sample(population, 2)
        temp_population.append(min(temp, key=lambda ind: ind.fitness_value))
    return temp_population
    
# ф-ция упорядоченного кроссинговера
def simple_crossover(parent_1, parent_2):
    i, j = randint(4, (LENGTH // 2) - 1), randint((LENGTH // 2) + 1, LENGTH - 6)
    child_1, child_2 = [-1]*29, [-1]*29
    child_1[i:j], child_2[i:j] = parent_1[i:j], parent_2[i:j]
    temp_1 = [num for num in parent_2[j:] + parent_2[:j] if num not in child_1]
    temp_2 = [num for num in parent_1[j:] + parent_1[:j] if num not in child_2]
    indices = list(range(j, LENGTH)) + list(range(0, i))
    for index, num in zip(indices, temp_1):
        child_1[index] = num
    for index, num in zip(indices, temp_2):
        child_2[index] = num
    return Individual(child_1), Individual(child_2)
    
# ф-ция случайной мутации
def mutate_gen(mutant):
    i, j = -1, -1
    while i == j:
        i, j = randrange(LENGTH), randrange(LENGTH)
    mutant[i], mutant[j] = mutant[j], mutant[i]
    
# создание начальной поауляции
population = population_creator(POPULATION_SIZE)
calculations(population)
    
optimums = []
    
# основной цикл генетического алгоритма
for i in range(NUM_GENERATIONS):    
    temp_population = selection_tournament(population)
    
    childs = []
    for j in range(POPULATION_SIZE // 2):
        parent_1, parent_2 = sample(temp_population, 2)
        if random() < P_CROSSOVER:
            child_1, child_2 = simple_crossover(parent_1, parent_2)
            for child in (child_1, child_2):
                if random() < P_MUTATION:
                    mutate_gen(child)
            childs.extend([child_1, child_2])

    calculations(childs)
    
    # пропорциональная редукция
    childs = sorted(childs, key=lambda x: x.fitness_value)[:AMOUNT_CHILDS]
    pop_size = POPULATION_SIZE - len(childs)
    population = sorted(population, key=lambda x: x.fitness_value)[:pop_size] + childs
    # определение самого приспособленного в популяции
    temp_optimum = min(population, key=lambda x: x.fitness_value)
    # добавление значения в коллекцию
    optimums.append((temp_optimum, i + 1))
    
best_value = min(optimums, key=lambda x: x[0].fitness_value)
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f'Время работы программы = {elapsed_time}')
print(f'Кол-во индивидуумов в популяции = {POPULATION_SIZE}')
print(f'Вероятность скрещивания = {P_CROSSOVER}')
print(f'Вероятность мутации = {P_MUTATION}')
print(f'Поколение в котором обнаруженно решение = {best_value[1]}')
print(f'Минимальная длина тура = {best_value[0].fitness_value}')
print(f'{best_value[0]}')
print('Длины оптимального тура = 9073')
print('[0, 27, 5, 11, 8, 25, 2, 28, 4, 20, 1, 19, 9, 3, 14, 17, 13, 16, 21, 10, 18, 24, 6, 22, 7, 26, 15, 12, 23]')

coordinate_x = np.array([x[n] for n in best_value[0] + [best_value[0][0]]])
coordinate_y = np.array([y[n] for n in best_value[0] + [best_value[0][0]]])

plt.plot(coordinate_x, coordinate_y)
# добавляем точки
plt.scatter(coordinate_x, coordinate_y, color='red', s=30, marker='o')
# отображаем сетку
plt.grid()
# показываем график
plt.show()