from optlear.mip import mip_wrapper


sample_horizontal_breakpoints = [0, 1, 2, 3]
sample_vertical_breakpoints = [0, 1, 4, 9]


class lambdaPWL(mip_wrapper.mipWrapper):

    def __init__(self, solver_package, pwl_model, verbose=False):

        self.solver_package = solver_package
        self.pwl_model = pwl_model
        self.verbose = verbose

    def _warn_breakpoints(self):
        """ Print a warning about the breakpoints """

        print("When dealing with k segments, make sure to have k + 1 breakpoints!")

        return None
        
    def set_vertical_breakpoints(self, vertical_breakpoints):
        """ Set the vertical breakpoints for the PWL approximation """

        self.warn_breakpoints()
        self.vertical_breakpoints = vertical_breakpoints

        return None

    def set_horizontal_breakpoints(self, horizontal_breakpoints):
        """ Set the vertical breakpoints for the PWL approximation """

        self.warn_breakpoints()
        self.horizontal_breakpoints = horizontal_breakpoints

        return None

    def name_lambda_variable(self, name_stem="h_", breakpoint_index, base_index=None):
        """ Name the lambda variable, giving it indices to distinguish it """

        name = name_stem
        if base_index is not None:
            name = name + f"{base_index}"
        name = name + f"{breakpoint_index}"

        return name

    def name_binary_variable(self, name_stem="w_", breakpoint_index, base_index=None):
        """ Name the binary variable, giving it indices to distinguish it """

        name = name_stem
        if base_index is not None:
            name = name + f"{base_index}"
        name = name + f"{breakpoint_index}"

        return name

    def build_lambda_variables(self, name_stem="h_", base_index=None):
        """ Build all of the necessary lambda variables, naming them as stated """

        variables = []
        for num in range(0, len(self.horizontal_breakpoints[:-1])):
            variable_name = self.name_lambda_variable(
                name_stem=name_stem,
                breakpoint_index=num,
                base_index=base_index,
            )
            variable = self.build_continuous_variable(0, 1, variable_name)
            variables.append(variable)

        return variables
