from optlearn import suppress

from concorde.tsp import TSPSolver


def solution_from_path(fname, solver=None, verbose=False):
    """ Load a problem and solve it """

    if verbose:
        solver = solver or TSPSolver.from_tspfile(fname)
        return solver.solve()
    else:
        with suppress.HiddenPrints():
            solver = solver or TSPSolver.from_tspfile(fname)
            return solver.solve()
        

    
    
