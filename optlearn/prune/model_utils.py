import joblib


class modelPersister():

    def __init__(self, model=None, function_names=None, threshold=None):
        
        self.model = model
        self.function_names = function_names
        self.threshold = threshold

    def set_metadata(self, meta_dict):
        """ Set all the given metadata """

        for key in meta_dict:
            self.key = meta_dict[key]

    def set_function_names(self, function_names):
        """ Set the function names """

        self.function_names = function_names

    def set_model(self, model):
        """ Set the model """

        self.model = model

    def set_threshold(self, threshold):
        """ Set the threshold """

        self.threshold = threshold
        
    def _check_model(self):
        """ Check if the model is given """

        if self.model is None:
            raise ValueError("No model set!")

    def _check_function_names(self):
        """ Check if the function names are given """

        if self.model is None:
            raise ValueError("No fucntion names set!")

    def _perform_checks(self):
        """ Perform the given checks """

        self._check_model()
        self._check_function_names()
        
    def _get_metadata(self, persist_dict):
        """ Get the metadata dictionary from the persist dictionary """

        return persist_dict["metadata"]

    def _get_function_names(self, meta_dict):
        """ Get the function names from the metadata dictioanry """

        return meta_dict["function_names"]

    def _get_model(self, persist_dict):
        """ Get the model from the persist dictionary """

        return persist_dict["model"]

    def _get_threshold(self, persist_dict):
        """ Get the threshold from the persist dictionary """

        return persist_dict["threshold"]
    
    def _build_meta_dict(self):
        """ Build the metadata dictionary """

        return {
            "function_names": self.function_names
            }

    def _build_persist_dict(self):
        """ Build the metadata dictionary """

        return {
            "model": self.model,
            "metadata": self._build_meta_dict(),
            "threshold": self.threshold,
            }
        
    def save(self, fname):
        """ Dump the model, saving metadata too """

        self._perform_checks()

        persist_dict = self._build_persist_dict()
        joblib.dump(persist_dict, fname)
        
    def load(self, fname):
        """ Load a model, loading the model and metadata """

        persist_dict = joblib.load(fname)
        self.set_model(self._get_model(persist_dict))
        self.set_metadata(self._get_metadata(persist_dict))
        self.set_function_names(self._get_function_names(self._get_metadata(persist_dict)))
        self.set_threshold(self._get_threshold(persist_dict))
        
        
