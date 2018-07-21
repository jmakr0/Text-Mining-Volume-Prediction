from src.utils.settings import Settings


class ModelBuilder:
    def __init__(self):
        self.required_inputs = []
        self.required_parameters = []

        self.inputs = {}
        self.parameters = {}

        settings = Settings()
        self.parameters = settings.get_network_parameters()

    def set_input(self, input_name, value):
        self.inputs[input_name] = value
        return self

    def set_parameter(self, parameter_name, value):
        self.parameters[parameter_name] = value
        return self

    def check_required(self):
        for required_input in self.required_inputs:
            if required_input not in self.inputs:
                raise MissingRequiredError('Missing required input: {}'.format(required_input))
        for required_parameter in self.required_parameters:
            if required_parameter not in self.parameters:
                raise MissingRequiredError('Missing required parameter: {}'.format(required_parameter))

    def __call__(self):
        raise NotImplementedError

    @property
    def model_identifier(self):
        raise NotImplementedError


class MissingRequiredError(ValueError):
    pass
