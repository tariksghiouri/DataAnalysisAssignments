# Example of calculating Euclidean distance
from math import sqrt

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Test distance function
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
row0 = dataset[0]
for row in dataset:
	distance = euclidean_distance(row0, row)
	print(distance)
 

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors


neighbors = get_neighbors(dataset, [5,3], 4)
for neighbor in neighbors:
 print(neighbor)
 
 
# Extract the classes from the data
classes = [sample[2] for sample in neighbors]

# Count the occurrences of each class
class_counts = {}
for c in classes:
    if c in class_counts:
        class_counts[c] += 1
    else:
        class_counts[c] = 1

# Find the most common class
max_count = 0
most_common_class = None
for c, count in class_counts.items():
    if count > max_count:
        max_count = count
        most_common_class = c

print("class:", most_common_class)