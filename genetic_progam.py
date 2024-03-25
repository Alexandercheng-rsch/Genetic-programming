import re
import math
from itertools import chain, repeat, product
from random import randint, random, choice, shuffle
from copy import deepcopy
import time
import numpy as np
import sys
import os
import random
import time
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import pickle
# import pandas as pd
dbg = False
term_regex = r'''(?mx)
    \s*(?:
        (?P<brackl>\()|
        (?P<brackr>\))|
        (?P<num>\-?\d+\.\d+|\-?\d+)|
        (?P<sq>"[^"]*")|
        (?P<s>[^(^)\s]+)
        )'''



def main1():

    question = None
    n = None
    expr = None
    m = None
    data = None
    x = None
    data = None 
    time_budget = None
    lambda_ = None


    # Loop through the arguments
    args = sys.argv[1:]  # Exclude the script name
    for i in range(len(args)):
        if args[i] == '-question':
            question = int(args[i + 1])  # Convert next item to int
        elif args[i] == '-n':
            n = int(args[i + 1])  # Convert next item to int
        elif args[i] == '-expr':
            expr = args[i + 1] 
        elif args[i] == '-x':
            x = args[i + 1]
        elif args[i] == '-m':
            m = int(args[i + 1])
        elif args[i] == '-data':
            data = args[i + 1]
        elif args[i] == '-time_budget':
            time_budget = int(args[i + 1])
        elif args[i] == '-lambda':
            lambda_ = int(args[i + 1])

    if question == 1:
        try:
            x_ = [float(i) for i in x.split()]
            data_handler = DataHandler(x_, n)
            parsed_expression = parse_sexp(expr)
            t = create_tree(parsed_expression)
            return print(t.evaluate(data_handler))
        except:
            print('Invalid expression')
            raise
    if question == 2:
        parsed_expression = parse_sexp(expr)
        t = create_tree(parsed_expression)
        assert os.path.isfile(data)
        x_ , y_ = reader(data)
        assert len(x_) == m and len(y_) == m
        assert all(len(i) == n for i in x_)
        return print(fitness(t,n,m,x_,y_))
    if question == 3:
        try:
            x_, y_ = reader(data)
            output = genetic_program(lambda_, n, m, x_, y_, time_budget)
            return print(output.to_s_expression())
        except:
            print('Invalid expression')
            raise
# #(crossover_param, mutation_param, tournament_param,pop_size,elite_fact , time_budget, x, y, n, m)
def dispatch(argument):
    if argument == 1:
        main1()
    elif argument == 2:
        main2()
    else:
        print('Invalid argument')
        sys.exit(1)

def main2():
    print('Using main 2')
    crossover_param = [ 0.7, 0.85, 0.9]
    mutation_param = [0.1, 0.2, 0.3]

    cross_dic, mut_dic = genetic_tuner(crossover_param, mutation_param)
    with open('cross_dic.pkl', 'wb') as f:
        pickle.dump(cross_dic, f)
    with open('mut_dic.pkl', 'wb') as f:
        pickle.dump(mut_dic, f)
    with open('results.txt', 'r') as f:
        result = f.read()


operators = {
    'add': lambda args, data_handler: args[0] + args[1],
    'mul': lambda args, data_handler: args[0]*args[1],
    'sub': lambda args, data_handler: args[0] - args[1],
    'log': lambda args, data_handler: math.log(args[0],2) if args[0] > 0 else 0,
    'div': lambda args, data_handler: (args[0]/args[1] if args[1] !=0 else 0),
    'sqrt' : lambda args, data_handler: math.sqrt(args[0]) if args[0] >= 0 else 0,
    'pow': lambda args, data_handler: args[0]**args[1] if (args[0] >= 0 or float(args[1]).is_integer()) and not (args[0] == 0 and args[1] < 0) else 0,
    'exp': lambda args, data_handler: math.exp(args[0]),
    'max' : lambda args, data_handler: max(args[0],args[1]),
    'ifleq' : lambda args, data_handler: args[2] if args[0] <= args[1] else args[3],
    'data' : lambda args, data_handler: (lambda j: data_handler.x[j] if data_handler.n != 0 else 0) (abs(math.floor(args[0])) % data_handler.n),
    'diff': lambda args, data_handler: (lambda i, j: data_handler.x[int(i)] - data_handler.x[int(j)])(abs(math.floor(float(args[0]))) % data_handler.n, abs(math.floor(float(args[1]))) % data_handler.n) if all(isinstance(arg, (int, float)) for arg in args) else 0,
    'avg': lambda args, data_handler: (lambda i, j: sum(data_handler.x[min(i,j):max(i,j)])/abs(j - i) if i != j else 0)(abs(math.floor(args[0])) % data_handler.n, abs(math.floor(args[1])) % data_handler.n) if data_handler.n != 0 else 0,
    }

terminals = {
    'data' : lambda args, data_handler: (lambda j: data_handler.x[j] if data_handler.n != 0 else 0) (abs(math.floor(args[0])) % data_handler.n),

}


functional_set = {'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'log': 1, 'sqrt': 1, 'pow': 2, 'exp': 1, 'max': 2, 'ifleq': 4, 'diff': 2, 'data': 1, 'avg': 2}


class DataHandler:
    def __init__(self, x, n):
        self.x = x
        self.n = n

class Tree:
    def __init__(self, value = None, left = None, right = None, middle1 = None, middle2 = None, arity = None, fitness = None):
        self.value = value
        self.left = left
        self.right = right 
        self.middle1 = middle1
        self.middle2 = middle2
        self.fitness = fitness
        self.arity = functional_set.get(self.value, 0)
        self.depth = 0 
        if self.value is not None:
            self.arity = functional_set.get(self.value, 0)
        else:
            self.arity = 0


    def to_s_expression(self):
        # Handle constants directly
        if isinstance(self.value, (int, float)):
            return str(self.value)
        if isinstance(self.left, (int, float)):
            return str(self.left)
        if isinstance(self.right, (int, float)):
            return str(self.right)
        if isinstance(self.middle1, (int, float)):
            return str(self.middle1)
        if isinstance(self.middle2, (int, float)):
            return str(self.middle2)
        if isinstance(self.value, tuple):
            return f"(data {self.value[1]})"

        # Construct expressions based on the number and type of children
        expressions = []
        if self.left:
            expressions.append(self.left.to_s_expression())
        if self.right:
            expressions.append(self.right.to_s_expression())
        if self.middle1:  # Assuming your structure might use middle1
            expressions.append(self.middle1.to_s_expression())
        if self.middle2:  # Assuming your structure might use middle2
            expressions.append(self.middle2.to_s_expression())

        # Combine expressions
        if expressions:
            return f"({self.value} {' '.join(expressions)})"
        else:
            # If no known conditions are met, return a representation of the value
            # or raise an error if this state is unexpected
            return f"({self.value})"

    def build_subtree(self):
        t = Tree()
        t.value = self.value
        # Check if `self.left` is a Tree instance before trying to build a subtree
        if isinstance(self.left, Tree):
            t.left = self.left.build_subtree()
        else:
            t.left = self.left  # Simply assign the value if it's an int/float

        # Do the same check for `self.right`, `self.middle1`, and `self.middle2`
        if isinstance(self.right, Tree):
            t.right = self.right.build_subtree()
        else:
            t.right = self.right

        if isinstance(self.middle1, Tree):
            t.middle1 = self.middle1.build_subtree()
        else:
            t.middle1 = self.middle1

        if isinstance(self.middle2, Tree):
            t.middle2 = self.middle2.build_subtree()
        else:
            t.middle2 = self.middle2

        return t

    def random_tree(self, method, max_depth, n, depth=0):
        if depth >= max_depth:
            # Terminal node
            if random.random() < 0.5:
                self.value = 'data'
                self.left = Tree(randint(0, n-1))
                self.right = None
                self.arity = 0
                self.depth = depth
            else:
                # Constant node
                self.value = randint(0, n-1)
                self.arity = 0
                self.depth = depth
        else:
            # Non-terminal node
            self.value = choice(list(operators.keys()))
            arity = functional_set[self.value]


            # Depending on arity, assign children recursively
            if arity >= 1:
                self.left = Tree()
                self.left.random_tree(method, max_depth, n, depth + 1)
                self.depth = depth
            if arity >= 2:
                self.right = Tree()
                self.right.random_tree(method, max_depth, n, depth + 1)
                self.depth = depth
            if arity >=3:  # Assuming max arity is 4 for any operator
                self.middle1 = Tree()
                self.middle1.random_tree(method, max_depth, n, depth + 1)
                self.middle2 = Tree()
                self.middle2.random_tree(method, max_depth, n, depth + 1)
                self.depth = depth

    def evaluate(self, data_handler):
        # Handle terminal 'data' node

        # Handle constant values directly
        if isinstance(self.value, (int, float)):
            return self.value

        # Handle operator nodes
        elif self.value in operators:

            # Collect arguments from child nodes
            if functional_set[self.value] == 1 and self.left is not None:
                args = operators[self.value]([self.left.evaluate(data_handler)], data_handler)
                return args
            
            
            if functional_set[self.value] == 2 and (self.left and self.right is not None):
                args = operators[self.value]([self.left.evaluate(data_handler), self.right.evaluate(data_handler)], data_handler)
                return args
            if functional_set[self.value] == 3 and (self.left and self.right and self.middle1 is not None):
                args = operators[self.value]([self.left.evaluate(data_handler), self.right.evaluate(data_handler),self.middle1.evaluate(data_handler)], data_handler)
                return args
            if functional_set[self.value] == 4 and (self.left and self.right and self.middle1 and self.middle2 is not None):
                args = operators[self.value]([self.left.evaluate(data_handler), self.right.evaluate(data_handler),self.middle1.evaluate(data_handler), self.middle2.evaluate(data_handler)], data_handler)
                return args

        # Unrecognized or malformed node
        return 0


    def size(self):
        if isinstance(self.value,(int, float)):
            return 0
        if self.left and not self.right:
            l = self.left.size() if isinstance(self.left, Tree) else 1
            return 1 + l
        if self.left and self.right:
            l = self.left.size() if isinstance(self.left, Tree) else 1
            r = self.right.size() if isinstance(self.right, Tree) else 1
            return 1 + l + r
        if self.left and self.right and self.middle1 and self.middle2:
            l = self.left.size() if isinstance(self.left, Tree) else 1
            r = self.right.size() if isinstance(self.right, Tree) else 1
            m1 = self.middle1.size() if isinstance(self.middle1, Tree) else 1
            m2 = self.middle2.size() if isinstance(self.middle2, Tree) else 1
            return 1 + l + r + m1 + m2

        



    def mutate(self, mutation_rate, max_depth, max_size, n):
        """
        Perform mutation on the tree ensuring the mutated tree does not exceed max_size.
        """
        attempts = 0
        max_attempts = 10  # Set a limit to prevent infinite loops
        nodes = self.get_random_node()  # Ensure nodes is always initialized

        while attempts < max_attempts:
            if random.random() < mutation_rate and nodes:
                node = random.choice(nodes)

                original_node = deepcopy(node)
                if  node.left:
                    node.left.random_tree('grow', 1, n)
                if  node.right:
                    node.right.random_tree('grow',1, n)
                if  node.middle1:
                    node.middle1.random_tree('grow',1,n)
                if node.middle2:
                    node.middle2.random_tree('grow',1,n)
                # Check if the mutated tree exceeds the max size
                if self.size() > max_size:
                    # Revert the mutation if the tree is too large
                    node = original_node
                    attempts += 1
                    continue
                break  # Mutation successful, break the loop

            # If mutation was not successful, or no nodes were found, try mutating child nodes
            if not nodes or attempts >= max_attempts:
                if self.left:
                    self.left.mutate(mutation_rate, max_depth, max_size, n)
                if self.right:
                    self.right.mutate(mutation_rate, max_depth, max_size, n)
                if self.middle1:
                    self.middle1.mutate(mutation_rate, max_depth, max_size, n)
                if self.middle2:
                    self.middle2.mutate(mutation_rate, max_depth, max_size, n)
                break  # Exit after trying to mutate child nodes







    def get_random_node(self, current_depth=0):
        """
        Randomly select a node in the tree at or below the specified depth.
        """
        nodes_at_depth = [self]  # Start with the current node
        MAX_DEPTH = 4  # Maximum depth to search for nodes
        # Recursively get nodes from children if they exist
        if self.left and current_depth < MAX_DEPTH:
            child_nodes = self.left.get_random_node(current_depth + 1)
            if child_nodes:  # Check if child_nodes is not None and not an empty list
                nodes_at_depth.extend(child_nodes)  # Extend the list with nodes from the left child

        if self.right and current_depth < MAX_DEPTH:
            child_nodes = self.right.get_random_node(current_depth + 1)
            if child_nodes:
                nodes_at_depth.extend(child_nodes)

        # Similar logic for middle1 and middle2, if they exist

        return nodes_at_depth
    def crossover_with(self, other, crossover_rate=0.8):
        """
        Perform crossover between this tree and another tree.
        """
        if not isinstance(other, Tree):
            return  # Ensure 'other' is a Tree instance

        # Select random nodes from each tree
        if random.random() < crossover_rate:
            nodes1 = self.get_random_node()
            nodes2 = other.get_random_node()

            # Randomly select one node from each list
            if nodes1 and nodes2:  # Check if both lists are not empty
                node1 = random.choice(nodes1)
                node2 = random.choice(nodes2)

                # Swap the selected nodes
                node1.value, node2.value = node2.value, node1.value
                node1.left, node2.left = node2.left, node1.left
                node1.right, node2.right = node2.right, node1.right
                node1.middle1, node2.middle1 = node2.middle1, node1.middle1
                node1.middle2, node2.middle2 = node2.middle2, node1.middle2


def create_tree(expr):
    t = Tree()
    if isinstance(expr, (float, int)):  # Directly set value for numbers
        t.value = expr
        return t

    op, *args = expr
    t.value = op
    t.arity = functional_set.get(t.value, 1)  # Get arity with a default of 0

    # Recursively create subtrees based on the arity and provided args
    if t.arity == 1:
        t.left = create_tree(args[0])
    elif t.arity == 2:
        t.left = create_tree(args[0])
        t.right = create_tree(args[1])
    elif t.arity == 4:
        t.left = create_tree(args[0])
        t.right = create_tree(args[1])
        t.middle1 = create_tree(args[2])
        t.middle2 = create_tree(args[3])

    return t




def selection(population, TOURNAMENT_SIZE):
    # Randomly select TOURNAMENT_SIZE individuals for the tournament
    tournament = [population[randint(0, len(population) - 1)] for _ in range(TOURNAMENT_SIZE)]

    # Directly find the individual with the best fitness without sorting
    best_individual = min(tournament, key=lambda x: x.fitness)

    return best_individual

def genetic_program(POP_SIZE, n, m, x, y,time_budget=60, MAX_DEPTH=1, TOURNAMENT_SIZE=3, mutation_rate=0.4, crossover_rate=0.9, elite_fact=0.1):
    # Initialize population
    population = spawn_pop(n, POP_SIZE, MAX_DEPTH, m, x, y)
    max_size = 8
    # Set the start time
    start_time = time.time()

    generation = 1
    while time.time() - start_time < time_budget:
        # Evaluate fitness of each individual
        data = []
        # Select parents for reproduction
        elites = math.floor(elite_fact * POP_SIZE)
        parents = []
        for _ in range(POP_SIZE):
            parent1 = selection(population, TOURNAMENT_SIZE)
            parent2 = selection(population, TOURNAMENT_SIZE)
            parents.append((parent1, parent2))

        # Create offspring through crossover and mutation

        offspring = []
        for parent1, parent2 in parents:
            offspring1 = deepcopy(parent1)
            offspring2 = deepcopy(parent2)
            offspring1.crossover_with(offspring2, crossover_rate)
            offspring1.mutate(mutation_rate, MAX_DEPTH,max_size, n)

            offspring1.fitness = fitness(offspring1, n, m, x, y, penalty=0, rate=0.1)
            offspring.append(offspring1)

        # Replace the least fit individuals with the offspring
        population = sorted(population + offspring, key=lambda x: x.fitness)[:elites]

        # Print the best individual in each generation
        best_individual = min(population, key=lambda x: x.fitness)
        data.append(best_individual.fitness)
        print(best_individual.to_s_expression())
        print(fitness(best_individual,n,m,x,y))
        print(best_individual.size())
        # Increment the generation counter
        generation += 1

    # Return the best individual from the final generation
    
    best_individual = min(population, key=lambda x: x.fitness)
    print(fitness(best_individual,n,m,x,y))
    return best_individual






def spawn_pop(n, pop_size, max_depth, m, x, y, ):
    population = []
    for _ in range(pop_size):
        method = random.choice(['full', 'grow'])
        tree = Tree()
        tree.random_tree(method, max_depth, n,)
        tree.fitness = fitness(tree, n, m, x, y, penalty=0)
        population.append(tree)
    return population


def fitness(expr, n, m, x, y, penalty=0, rate = 0.01):
    count = 0
    try:
        for i in range(m):  
      
            data_handler = DataHandler(x[i], n)
            evaluated_expr = expr.evaluate(data_handler)
            diff = y[i] - evaluated_expr
            squared_diff = diff**2
            count += squared_diff   
        comp = complexity_penalty(expr, rate)

    
    except OverflowError:
        return float('inf')
    if penalty == 0:
        return count/m
    else:
        return count/m + comp



def complexity_penalty(tree, penalty_rate=0):
    min_complexity = 7 # Set a minimum desired complexity
    complexity = tree.size()  
    
    if complexity > min_complexity:
        return (min_complexity - complexity) * penalty_rate  # penalty_rate is a constant defining how harsh the penalty is
    else:
        return 0


'''
 Reads the data and turns it into an array
'''   
def reader(filename):

	with open(filename) as f: data = f.readlines()

	data = [x.split('\t') for x in data]

	x = [[float(j.strip()) for j in i[:len(i)-1]] for i in data]
	y = [float(i[len(i)-1].strip()) for i in data]

	return x, y



def parse_sexp(expression):
    stack = []
    out = []
    if dbg: print("%-6s %-14s %-44s %-s" % tuple("term value out stack".split()))
    for termtypes in re.finditer(term_regex, expression):
        term, value = [(t,v) for t,v in termtypes.groupdict().items() if v][0]
        if dbg: print("%-7s %-14s %-44r %-r" % (term, value, out, stack))
        if   term == 'brackl':
            stack.append(out)
            out = []
        elif term == 'brackr':
            assert stack, "Trouble with nesting of brackets"
            tmpout, out = out, stack.pop(-1)
            out.append(tmpout)
        elif term == 'num':
            v = float(value)
            if v.is_integer(): v = int(v)
            out.append(v)
        elif term == 'sq':
            out.append(value[1:-1])
        elif term == 's':
            out.append(value)
        else:
            raise NotImplementedError("Error: %r" % (term, value))
    assert not stack, "Trouble with nesting of brackets"
    return out[0]         

def genetic_tuner(crossover_param, mutation_param):
    data = 'cetdl1772small.dat'
    n = 13
    m = 999
    x, y = reader(data)
    cross_result = {}
    mutation_result = {}

    for keys in crossover_param:
        cross_result[keys] = []
    for keys in mutation_param:
        mutation_result[keys] = []

    with ProcessPoolExecutor() as executor:
        # Perform crossover computations in parallel
        for j in range(100):
            crossover_tasks = []
            for param in crossover_param:
                task = executor.submit(genetic_program, 100, n, m, x, y, time_budget=10, crossover_rate=param)
                crossover_tasks.append((param, task))
            
            for param, task in crossover_tasks:
                result = task.result()
                fitness_value = fitness(result, n, m, x, y)
                cross_result[param].append(fitness_value)  # Append fitness value to the list for the corresponding parameter
            print(f'Completed {j} iterations')
        # Perform mutation computations in parallel
        for j in range(100):
            mutation_tasks = []
            for param in mutation_param:
                task = executor.submit(genetic_program, 100, n, m, x, y, time_budget=10, mutation_rate=param)
                mutation_tasks.append((param, task))
            for param, task in mutation_tasks:
                result = task.result()
                fitness_value = fitness(result, n, m, x, y)
                mutation_result[param].append(fitness_value)  # Append fitness value to the list for the corresponding parameter
            print(f'2nd Completed {j} iterations')
    return cross_result, mutation_result        

if __name__ == '__main__':
    main1()
# if __name__ == '__main__':
        #  main2() #Uncomment for tuning 




data = 'cetdl1772small.dat'
x, y = reader(data)
m = 999
n = 13
genetic_program(100,n,m,x,y,time_budget=200)




# import matplotlib.pyplot as plt

# obj = pd.read_pickle(r'cross_dic.pkl')
# data = [obj[0.7], obj[0.85], obj[0.9]]
# labels = ['0.7', '0.85', '0.9']
# # [ 0.7, 0.85, 0.9]
# plt.boxplot(data, labels=labels)
# plt.title('Crossover vs Fitness')
# plt.show()

# m = 999
# n = 13
# storage = []
# for i in range(20):
#     result = genetic_program(100, 13, 999, x, y, time_budget=10, crossover_rate=0.85, mutation_rate=0.1, MAX_DEPTH = 2, TOURNAMENT_SIZE=3)
#     storage.append(fitness(result, n, m, x, y, penalty=0, rate=0))
#     print(fitness(result, n, m, x, y, penalty=0, rate=0))

# with open('result.pkl', 'wb') as f:
#     pickle.dump(storage, f)
# import matplotlib.pyplot as plt
# obj = pd.read_pickle(r'result.pkl')
# plt.boxplot(obj)
# plt.title('Crossover = 0.85 and mutation rate = 0.1')
# plt.show()
