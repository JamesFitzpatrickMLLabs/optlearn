class binaryFixer():
    def __init__(self, model, edge_fn=None, vertex_fn=None):
        self.model = model
        self.feature_fn = None

    def set_edge_fn(self, edge_fn):
        """ Set function to compute features for an edge, given graph and that edge """
        
        self.edge_fn = edge_fn

    def set_vertex_fn(self, vertex_fn):
        """ Set function to compute features for outward egdes of a vertex, given graph
        and that vertex  """
        
        self.vertex_fn = vertex_fn

    def predict_edge(self, graph, edge):
        """ Make a prediction for a given edge """

        return self.predict_feature(self.edge_fn(graph, edge))

    def predict_edges(self, graph, edges):
        """ Make predictions for a given edges """
        
        return [self.predict_edge(graph, edge) for edge in edges]

    def predict_vertex(self, graph, vertex):
        """ Make prediction for all outward edges associated with a given vertex """

        return self.predict_features(self.vertex_fn(graph, vertex))
        
    def predict_feature(self, feature):
        """ Given a computed feature, make a prediction on it """
        
        return self.model.predict(feature)

    def predict_features(self, features):
        """ Given computed featuresm make predictions on them """
        
        return [self.predict_feature(feature) for feature in features]

    def fix_weights(self, weights, labels):
        """ Alter the values of the edge weights that are to be pruned so that they are large """

        if not type(weights) == np.ndarray:
            weights = np.array(weights)
        if not type(labels) == np.ndarray:
            weights = np.array(labels)
        if len(weights) != len(labels):
            raise ValueError("len(weights) and len(labels) should be equal!")
        weights[labels == 0] = 9999999999
        return weights.tolist()

