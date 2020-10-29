from optlearn import cc_solve
from optlearn import io_utils
from optlearn import graph_utils
from optlearn import experiment_utils

from optlearn.mip import mip_model


files = experiment_utils.list_files_recursive("/home/james/Data/MATHILDA/tsp/")

solution_gaps = []


for num, file in enumerate(files):
    object = io_utils.optObject().read_from_file(file)
    graph = object.get_graph()
    graph = graph_utils.delete_self_weights(graph)
    graph = graph.to_undirected()

    problem = mip_model.tspProblem(graph=graph,
                                   var_type="binary",
                                   formulation="dantzig",
                                   solver="xpress",
                                   verbose=True
    )
    problem.solve()

    mip_objective = problem.get_objective_value()
    cc_objective = cc_solve.solution_from_path(file).optimal_value

    solution_gaps.append(int(mip_objective - cc_objective))

    
