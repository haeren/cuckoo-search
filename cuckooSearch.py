import numpy as np
import math

# Input Parameters
dimension = 2           # Number of the objection function's parameters
lowerBound = [-10, -10] # Lower bound of the search domain
upperBound = [10, 10]   # Upper bound of the search domain
population = 15         # Number of solutions
discoveryRate = 0.25    # Discovery rate of the cuckoo eggs (Pa)
maxGeneration = 1500    # Number of iterations

# Objection Function
def objFunc(params):
    x1 = params[0]
    x2 = params[1]
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 -5)**2
         
# Generate Initial Population Randomly
nests = np.empty((population, dimension))
fitnesses = np.empty(population)
for i in range(population):
    for j in range(dimension):
        nests[i,j] = lowerBound[j] + np.random.uniform() * (upperBound[j] - lowerBound[j]) # x = L + rand*(U - L)
    fitnesses[i] = objFunc(nests[i])

beta = 3/2
sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) \
         / (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    
# Cuckoo Search Main Loop
for t in range(maxGeneration):
    bestIndex = np.argmin(fitnesses)    # Index of minimum fitness
    bestFitness = np.amin(fitnesses)    # Value of minimum fitness
    bestNest = nests[bestIndex]         # Solution corresponding to minimum fitness
    
    # Levy flight
    for i in range(population):
        x = nests[i]                                # x = i th nest (solution)
        u = np.random.normal(size=len(x)) * sigma   # u = randn * sigma u
        v = np.random.normal(size=len(x))           # v = randn * sigma v, but sigma v = 1
        step = u / np.power(np.abs(v), (1 / beta))
        xNew = x + np.random.normal(size=len(x)) * 0.01 * step * (x - bestNest)
        
        # Check Bounds
        for j in range(len(xNew)):
            if xNew[j] > upperBound[j]:
                xNew[j] = upperBound[j]
            elif xNew[j] < lowerBound[j]:
                xNew[j] = lowerBound[j]
                
        # Greedy Selection
        fNew = objFunc(xNew)    # fNew: Fitness of xNew
        if fNew < fitnesses[i]:
            nests[i] = xNew
            fitnesses[i] = fNew
    
    # Replace Some Nests
    # This time, we calculate all new solutions in one step by using the whole matrix instead of iterating over rows (solutions)
    eggStatuses = (np.random.uniform(size=np.shape(nests)) < discoveryRate).astype(int)  # To check if an egg is discovered by host bird
    stepSize = np.random.uniform(size=np.shape(nests)) * (np.random.permutation(nests) - np.random.permutation(nests))   # Multiplying each argument with a different random number
    nestsNew = nests + stepSize * eggStatuses   # Only the discovered eggs will be replaced
    
    # Check Bounds
    for i in range(len(nestsNew)):
        for j in range(len(nestsNew[i])):
            if nestsNew[i,j] > upperBound[j]:
                nestsNew[i,j] = upperBound[j]
            elif nestsNew[i,j] < lowerBound[j]:
                nestsNew[i,j] = lowerBound[j]
                
    for i in range(len(nestsNew)):
        fNew = objFunc(nestsNew[i]) # fNew: Fitness of nestsNew's i th row
        if fNew < fitnesses[i]:
            nests[i] = nestsNew[i]
            fitnesses[i] = fNew
            
    # Print the Best Nest at the End of Iteration
    bestIndex = np.argmin(fitnesses)    # Index of minimum fitness
    bestFitness = np.amin(fitnesses)    # Value of minimum fitness
    bestNest = nests[bestIndex]         # Solution corresponding to min. fitness
    print('Iteration: {}, Solution: {}, Fitness: {:.20f}'.format(t+1, bestNest, bestFitness))
