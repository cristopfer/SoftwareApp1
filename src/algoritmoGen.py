import random

class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_length, max_generations, mutation_probability, a3, a2, a1, a0):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.max_generations = max_generations
        self.mutation_probability = mutation_probability
        self.a3 = a3
        self.a2 = a2
        self.a1 = a1
        self.a0 = a0

    def f(self,x):
        return self.a3 * x ** 3 + self.a2 * x ** 2 + self.a1 * x + self.a0

    ##def binary_to_decimal(self, binary):
        ##return int(binary, 2)
    
    def discrete_binary_to_decimal(self, binary):
        discrete_value = int(binary, 2)
        max_discrete = (1 << self.chromosome_length) - 1
        decimal_value = discrete_value * (63 / max_discrete)   
        return decimal_value

    ##def decimal_to_binary(self, n):
        ##return format(n, f'0{self.chromosome_length}b')

    def decimal_to_discrete_binary(self, n):
        max_discrete = (1 << self.chromosome_length) - 1 
        discrete_value = round(n * (max_discrete / 63))
        discrete_value = max(0, min(max_discrete, discrete_value))
        return format(discrete_value, f'0{self.chromosome_length}b')

    def generate_chromosome(self):
        return ''.join(random.choice('01') for _ in range(self.chromosome_length))

    def evaluate_population(self, population):
        return [(chromosome, self.f(self.discrete_binary_to_decimal(chromosome))) for chromosome in population]

    def select_parents(self, evaluated_population):
        sorted_evaluated = sorted(evaluated_population, key=lambda x: x[1], reverse=True)
        return [sorted_evaluated[0][0], sorted_evaluated[1][0]]

    def crossover(self, parent1, parent2):
        point = random.randint(1, self.chromosome_length - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    def mutate(self, chromosome):
        chromosome = list(chromosome)
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_probability:
                chromosome[i] = '0' if chromosome[i] == '1' else '1'
        return ''.join(chromosome)

    def run(self):
        population = [self.generate_chromosome() for _ in range(self.population_size)]
        best_overall = ("", -float("inf"))
        historial = []

        for generation in range(1, self.max_generations + 1):
            evaluated = self.evaluate_population(population)
            best_gen = max(evaluated, key=lambda x: x[1])

            historial.append({
                'generacion': generation,
                'gen': best_gen[0],
                'x': self.discrete_binary_to_decimal(best_gen[0]),
                'fx': float(best_gen[1])  
            })

            if best_gen[1] > best_overall[1]:
                best_overall = best_gen

            parents = self.select_parents(evaluated)

            children = []
            while len(children) < self.population_size:
                p1, p2 = random.choices(parents, k=2)
                c1, c2 = self.crossover(p1, p2)
                children.extend([self.mutate(c1), self.mutate(c2)])
            population = children[:self.population_size]

        return best_overall[1], historial