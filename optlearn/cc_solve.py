from concorde.tsp import TSPSolver


def solution_from_path(fname, solver=None, verbose=False):
    """ Load a problem and solve it """

    solver = solver or TSPSolver.from_tspfile(fname)
    return solver.solve(verbose=verbose)

    
    
