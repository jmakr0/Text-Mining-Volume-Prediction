class ModelBuilder:
    def __init__(self):
        self.required_inputs = []
        self.required_parameters = []

        self.default_inputs = {}
        self.default_parameters = {}

        self.inputs = {}
        self.parameters = {}

    def set_input(self, input_name, value):
        self.inputs[input_name] = value
        return self

    def set_parameter(self, parameter_name, value):
        self.parameters[parameter_name] = value
        return self

    def check_required(self):
        for required_input in self.required_inputs:
            if required_input not in self.inputs:
                raise MissingRequiredError
        for required_parameter in self.required_parameters:
            if required_parameter not in self.parameters:
                raise MissingRequiredError

    def set_defaults(self):
        for default_input, value in self.default_inputs.items():
            if default_input not in self.inputs:
                self.inputs[default_input] = value
        for default_parameter, value in self.default_parameters.items():
            if default_parameter not in self.parameters:
                self.parameters[default_parameter] = value

    def prepare_building(self):
        self.check_required()
        self.set_defaults()

    def get_model_description(self):
        description = ""

        description += "required_inputs:\n"
        description += str(self.required_inputs) + "\n"

        description += "required_parameters:\n"
        description += str(self.required_parameters) + "\n"

        description += "default_inputs:\n"
        description += str(self.default_parameters) + "\n"

        description += "default_parameters:\n"
        description += str(self.default_parameters) + "\n"

        description += "inputs:\n"
        description += str(self.inputs) + "\n"

        description += "parameters:\n"
        description += str(self.parameters) + "\n"

        return description

    def __call__(self):
        raise NotImplementedError


class MissingRequiredError(ValueError):
    pass
