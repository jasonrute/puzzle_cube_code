"""
A user-facing wrapper around the neural network models for solving the cube.
"""

import models
from typing import Optional


class CubeModel:
    _model = None  # type: Optional[models.BaseModel]

    def __init__(self):
        pass

    def load_from_config(self, filepath: Optional[str] = None) -> ():
        """
        Build a model from the config file settings.
        :param filepath: Optional string to filepath of model weights.
                         If None (default) then it will load based on the config file.
        """
        import config

        if filepath is None:
            assert False, "Fill in this branch"

        self.load(config.model_type, filepath, **config.model_kwargs)

    def load(self, model_type: str, filepath: str, **kwargs) -> ():
        """
        Build a model.
        :param model_type: The name of the model class in models.py
        :param filepath: The path to the model weights.
        :param kwargs: Key word arguements for initializing the model class (the one given by model_type).
        """
        model_constructor = models.__dict__[model_type]  # get model class by name
        self._model = model_constructor(**kwargs)

        self._model.build()
        self._model.load_from_file(filepath)

    def _function(self):
        assert (self._model is not None), "No model loaded"
        return self._model.function
