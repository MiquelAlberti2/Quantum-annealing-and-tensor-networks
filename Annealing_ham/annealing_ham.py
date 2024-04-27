N = 3
values = [[0] * (N-i) for i in range(N)]

# Assigning profit of each item
values[0][0] = 10  # Quadratic coefficient for item 1
values[0][1] = 5   # Quadratic coefficient for item 1 and item 2
values[0][2] = 3   # Quadratic coefficient for item 1 and item 3
values[1][1] = 20  # Quadratic coefficient for item 2
values[1][2] = 20  # Quadratic coefficient for item 2 and item 3
values[2][2] = 30  # Quadratic coefficient for item 3

# Assigning weight of each item
weights = [10, 20, 30]

