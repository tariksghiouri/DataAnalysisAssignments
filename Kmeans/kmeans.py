from math import sqrt

def euclidean_distance(x, y):
    return sqrt(sum((px - py)**2  for px, py in zip(x,y)))
    
print(euclidean_distance((1,9),(2,5)))