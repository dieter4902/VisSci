import sys
import numpy as np


def knapsack(items, maxweight):
    lookup = np.zeros((maxweight + 1, len(items) + 1))

    for i, (value, weight) in enumerate(items):
        i += 1
        for capacity in range(maxweight + 1):
            if weight > capacity:
                lookup[capacity, i] = lookup[capacity, i - 1]
            else:
                temp1 = lookup[capacity, i - 1]
                temp2 = lookup[capacity - weight, i - 1] + value
                lookup[capacity, i] = max(temp1, temp2)
    selected_items = []

    i = len(items)
    j = maxweight
    while i > 0:
        if lookup[j, i] != lookup[j, i - 1]:
            selected_items.append(items[i - 1])
            j -= items[i - 1][1]
        i -= 1
    return lookup[maxweight, len(items)], selected_items


maxweight = 7
items = [(16, 2), (19, 3), (23, 4), (28, 5)]
value, selected_items = knapsack(items, maxweight)
print("bagworth:", value)
print("selected items:", selected_items)
