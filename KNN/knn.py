from math import sqrt

#  Euclidean distance 
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

dataset = [[3, 3, 0],
           [1, 2, 0],
           [3, 4, 0],
           [1, 2, 0],
           [3, 3, 0],
           [8, 3, 1],
           [5, 2, 1],
           [7, 2, 1],
           [9, 0, 1],
           [8, 4, 1]]
row0 = dataset[0]
for row in dataset:
	distance = euclidean_distance(row0, row)
	print(distance)
 

# similar neighbors
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
 
 
# Extract the classes
classes = [sample[2] for sample in neighbors]

# Count the occurrences 
class_counts = {}
for c in classes:
    if c in class_counts:
        class_counts[c] += 1
    else:
        class_counts[c] = 1

# most common class and the  classification
max_count = 0
most_common_class = None
for c, count in class_counts.items():
    if count > max_count:
        max_count = count
        most_common_class = c

print("class:", most_common_class)