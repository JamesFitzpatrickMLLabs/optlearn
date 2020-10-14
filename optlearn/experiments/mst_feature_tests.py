import numpy as np

import matplotlib.pyplot as plt

from optlearn import experiment_utils
from optlearn import graph_utils
from optlearn import io_utils
from optlearn import cc_solve

from optlearn.mst import mst_model


files = experiment_utils.list_files_recursive("/home/james/Data/MATHILDA/tsp/")

model = mst_model.mstSparsifier()

ones, twos, threes, fours, fives = [], [], [], [], []

for num, file in enumerate(files[:30]):
    object = io_utils.optObject().read_from_file(file)
    graph = object.get_graph()
    graph = graph_utils.delete_self_weights(graph)
    graph = graph.to_undirected()

    solution = cc_solve.solution_from_path(file)
    edges = graph_utils.get_tour_edges(solution.tour)
    if np.min(solution.tour) < 1 and np.min(graph.nodes) > 0:
        edges += 1
    edges =  model.tuples_to_strings(edges)
    
    features = model.run_sparsify(graph, 15)
    features = [model.tuples_to_strings(item) for item in features]

    one = np.sum([edge in features[0] for edge in edges])
    two = np.sum([edge in features[1] for edge in edges]) + one
    three = np.sum([edge in features[2] for edge in edges]) + two
    four = np.sum([edge in features[3] for edge in edges]) + three
    five = np.sum([edge in features[4] for edge in edges]) + four
    length = len(solution.tour)
    
    ones.append(one/length)
    twos.append(two/length)
    threes.append(three/length)
    fours.append(four/length)
    fives.append(five/length)

    print("Processed {} of : 1330".format(num))


bins = 40

plt.figure(figsize=(20,10))
plt.title("Optimal Edges Captured after k=1 MWST Extraction", fontsize=20)
plt.grid()
plt.ylabel("Fraction of Problem Instances", fontsize=15)
plt.xlabel("Fraction of Optimal Edges", fontsize=15)
a, b = np.histogram(ones, bins=bins)
a = a/ 1300
plt.bar(b[1:], a, np.diff(b)[0], color="purple", linewidth=2, edgecolor="black")

plt.figure(figsize=(20,10))
plt.title("Optimal Edges Captured after k=2 MWST Extractions", fontsize=20)
plt.grid()
plt.ylabel("Fraction of Problem Instances", fontsize=15)
plt.xlabel("Fraction of Optimal Edges", fontsize=15)
a, b = np.histogram(twos, bins=bins)
a = a/ 1300
plt.bar(b[1:], a, np.diff(b)[0], color="green", linewidth=2, edgecolor="black")

plt.figure(figsize=(20,10))
plt.title("Optimal Edges Captured after k=3 MWST Extractions", fontsize=20)
plt.grid()
plt.ylabel("Fraction of Problem Instances", fontsize=15)
plt.xlabel("Fraction of Optimal Edges", fontsize=15)
a, b = np.histogram(threes, bins=bins)
a = a/ 1300
plt.bar(b[1:], a, np.diff(b)[0], color="red", linewidth=2, edgecolor="black")

plt.figure(figsize=(20,10))
plt.title("Optimal Edges Captured after k=4 MWST Extractions", fontsize=20)
plt.grid()
plt.ylabel("Fraction of Problem Instances", fontsize=15)
plt.xlabel("Fraction of Optimal Edges", fontsize=15)
a, b = np.histogram(fours, bins=bins)
a = a/ 1300
plt.bar(b[1:], a, np.diff(b)[0], color="orange", linewidth=2, edgecolor="black")

plt.figure(figsize=(20,10))
plt.title("Optimal Edges Captured after k=5 MWST Extractions", fontsize=20)
plt.grid()
plt.ylabel("Fraction of Problem Instances", fontsize=15)
plt.xlabel("Fraction of Optimal Edges", fontsize=15)
a, b = np.histogram(fives, bins=bins)
a = a/ 1300
plt.bar(b[1:], a, np.diff(b)[0], color="blue", linewidth=2, edgecolor="black")


names = ["CLKeasy",
         "hardCLK-easyLKCC",
         "random",
         "LKCCeasy",
         "easyCLK-hardLKCC",
         "LKCChard",
         "CLKhard"
]


plt.figure(figsize=(20,10))
plt.title("Optimal Edges Captured after k=1 MWST Extraction", fontsize=20)
plt.plot(ones, linewidth=0.5, color="purple")
plt.ylabel("Fraction of Optimal Edges Captured", fontsize=20)
plt.xlabel("MATILDA Problem Type", fontsize=20)
plt.grid()
for i in range(8):
    plt.axvline(x=190*i, linestyle="--", color="black", linewidth=2)
plt.xticks([num * 190 + 95 for num in range(7)], names, fontsize=12)

plt.figure(figsize=(20,10))
plt.title("Optimal Edges Captured after k=2 MWST Extractions", fontsize=20)
plt.plot(twos, linewidth=0.5, color="green")
plt.ylabel("Fraction of Optimal Edges Captured", fontsize=20)
plt.xlabel("MATILDA Problem Type", fontsize=20)
plt.grid()
for i in range(8):
    plt.axvline(x=190*i, linestyle="--", color="black", linewidth=2)
plt.xticks([num * 190 + 95 for num in range(7)], names, fontsize=12)

plt.figure(figsize=(20,10))
plt.title("Optimal Edges Captured after k=3 MWST Extractions", fontsize=20)
plt.plot(threes, linewidth=0.5, color="red")
plt.ylabel("Fraction of Optimal Edges Captured", fontsize=20)
plt.xlabel("MATILDA Problem Type", fontsize=20)
plt.grid()
for i in range(8):
    plt.axvline(x=190*i, linestyle="--", color="black", linewidth=2)
plt.xticks([num * 190 + 95 for num in range(7)], names, fontsize=12)

plt.figure(figsize=(20,10))
plt.title("Optimal Edges Captured after k=4 MWST Extractions", fontsize=20)
plt.plot(fours, linewidth=0.5, color="orange")
plt.ylabel("Fraction of Optimal Edges Captured", fontsize=20)
plt.xlabel("MATILDA Problem Type", fontsize=20)
plt.grid()
for i in range(8):
    plt.axvline(x=190*i, linestyle="--", color="black", linewidth=2)
plt.xticks([num * 190 + 95 for num in range(7)], names, fontsize=12)

plt.figure(figsize=(20,10))
plt.title("Optimal Edges Captured after k=5 MWST Extractions", fontsize=20)
plt.plot(fives, linewidth=0.5, color="blue")
plt.ylabel("Fraction of Optimal Edges Captured", fontsize=20)
plt.xlabel("MATILDA Problem Type", fontsize=20)
plt.grid()
for i in range(8):
    plt.axvline(x=190*i, linestyle="--", color="black", linewidth=2)
plt.xticks([num * 190 + 95 for num in range(7)], names, fontsize=12)


files = experiment_utils.list_files_recursive("/home/james/Data/TSPLIB/TSP/")

model = mst_model.mstSparsifier()


ones, twos, threes, fours, fives = [], [], [], [], []

for num, file in enumerate(files):
    object = io_utils.optObject().read_from_file(file)
    graph = object.get_graph()
    graph = graph_utils.delete_self_weights(graph)
    graph = graph.to_undirected()

    solution = cc_solve.solution_from_path(file)
    edges = graph_utils.get_tour_edges(solution.tour)
    if np.min(solution.tour) < 1 and np.min(graph.nodes) > 0:
        edges += 1
    edges =  model.tuples_to_strings(edges)
    
    features = model.run_sparsify(graph, 15)
    features = [model.tuples_to_strings(item) for item in features]

    one = np.sum([edge in features[0] for edge in edges])
    two = np.sum([edge in features[1] for edge in edges]) + one
    three = np.sum([edge in features[2] for edge in edges]) + two
    four = np.sum([edge in features[3] for edge in edges]) + three
    five = np.sum([edge in features[4] for edge in edges]) + four
    length = len(solution.tour)
    
    ones.append(one/length)
    twos.append(two/length)
    threes.append(three/length)
    fours.append(four/length)
    fives.append(five/length)

    print("Processed {} of : {}".format(num, len(files)))
