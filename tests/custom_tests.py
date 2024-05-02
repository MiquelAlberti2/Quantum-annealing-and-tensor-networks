import random 

def easy_test():
    N = 3
    values = [[0] * (i+1) for i in range(N)]

    # Assigning profit of each item
    values[0][0] = 60 
    values[1][0] = 30
    values[1][1] = 100   
    values[2][0] = 2  
    values[2][1] = 5  
    values[2][2] = 120  

    # Assigning weight of each item
    weights = [10, 20, 30]

    W_capacity = 50

    return N, values, weights, W_capacity

def random_test(N):
    values = [[0] * (i+1) for i in range(N)]

    # Assigning random profit for each item
    for i in range(N):
        for j in range(i + 1):
            values[i][j] = random.randint(1, 100)

    # Assigning random weight for each item
    weights = [random.randint(1, 100) for _ in range(N)]

    # Setting knapsack capacity
    W_capacity = random.randint(1, 100)

    return N, values, weights, W_capacity