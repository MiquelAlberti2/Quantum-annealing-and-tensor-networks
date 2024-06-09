import random 


def super_easy_test():
    N = 2
    values = [[0] * (i+1) for i in range(N)]

    # Assigning profit of each item
    values[0][0] = 60 
    values[1][0] = 30
    values[1][1] = 100   

    # Assigning weight of each item
    weights = [10, 20]

    W_capacity = 15

    return N, values, weights, W_capacity


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

def medium_test():
    N = 4
    values = [[0] * (i+1) for i in range(N)]

    # Assigning profit of each item
    values[0][0] = 58 
    values[1][0] = 62
    values[2][0] = 11
    values[3][0] = 88
    values[1][1] = 44
    values[2][1] = 11
    values[3][1] = 49 
    values[2][2] = 42
    values[3][2] = 29
    values[3][3] = 61

    # Assigning weight of each item
    weights = [29, 78, 36, 52]

    W_capacity = 114

    return N, values, weights, W_capacity

# --------------------------------------------------------

def random_test(N, mult = 3):
    values = [[0] * (i+1) for i in range(N)]

    # Assigning random profit for each item
    for i in range(N):
        for j in range(i + 1):
            values[i][j] = random.randint(1, 100)

    # Assigning random weight for each item
    weights = [random.randint(1, 100) for _ in range(N)]

    # Setting knapsack capacity
    W_capacity = random.randint(1, int(100*(N/mult)))

    return N, values, weights, W_capacity
