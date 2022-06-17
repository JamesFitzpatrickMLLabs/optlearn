import time
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class linearNodeEncoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(linearNodeEncoder, self).__init__()

        self.linear_encoder = nn.Linear(input_size, hidden_size)

    def forward(self, input):

        output = self.linear_encoder(input)
        
        return output


class singleHeadedNodeEncoder(nn.Module):

    def __init__(self, input_size, query_size, key_size, value_size):
        super(singleHeadedNodeEncoder, self).__init__()
        
        self.input_size = input_size
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        
        self.query_projector = nn.Linear(input_size, query_size, bias=False)
        self.value_projector = nn.Linear(input_size, value_size, bias=False)
        self.key_projector = nn.Linear(input_size, key_size, bias=False)

    def compute_compatibilities(self, node_queries, node_keys, mask=None):
        """ Compare the node queries with the object keys, with optional masking """
        
        compatibilities = torch.einsum("kij,klm->kil", node_queries, node_keys)
        compatibilities = compatibilities / (self.key_size) ** 0.5
        if mask is not None:
            if mask.shape != compatibilities.shape:
                raise Exception("Mask tensor must be the same shape as compatibilities tensor!")
            compatibilities[mask] = - torch.inf()
            
        return compatibilities

    def compute_attention_weights(self, compatibilities):
        """ Softmax the compatibility scores to obtain the attention weights """

        attention_weights = torch.softmax(compatibilities, axis=-1)

        return attention_weights

    def compute_ouput_values(self, node_values, attention_weights):
        """ Take a convex combination of the node values to get the attention values """

        output_values = torch.einsum("kij,kil->kij", node_values, attention_weights)
                
        return output_values

    def forward(self, node_embeddings, mask=None):

        node_queries = self.query_projector(node_embeddings)
        node_values = self.value_projector(node_embeddings)
        node_keys = self.key_projector(node_embeddings)
        
        compatibilities = self.compute_compatibilities(node_queries, node_keys, mask)
        attention_weights = self.compute_attention_weights(compatibilities)
        output_values = self.compute_ouput_values(node_values, attention_weights)
        
        return output_values


class singleHeadedContextNodeEncoder(nn.Module):

    def __init__(self, context_size, query_size, key_size, value_size):
        super(singleHeadedContextNodeEncoder, self).__init__()

        self.context_size = context_size
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        
        self.context_query_projector = nn.Linear(context_size, query_size, bias=False)
        self.value_projector = nn.Linear(int(context_size / 3), value_size, bias=False)
        self.key_projector = nn.Linear(int(context_size / 3), key_size, bias=False)
        
    def compute_compatibilities(self, context_query, node_keys, mask=None):
        """ Compute the compatibility of the context query with the other node keys """
        
        compatibilities = torch.einsum("kj,kij->ki", context_query, node_keys)
        compatibilities = compatibilities / (self.key_size) ** 0.5
        if mask is not None:
            if mask.shape != compatibilities.shape:
                raise Exception("Mask tensor must be the same shape as compatibilities tensor!")
            compatibilities[mask] = - torch.inf()

        return compatibilities

    def compute_attention_weights(self, compatibilities):
        """ Take a softmax over the compatibilities to get the attention weights """

        attention_weights = torch.softmax(compatibilities, -1)

        return attention_weights

    def compute_context_embedding(self, node_values, attention_weights):
        """ Compute the context embedding as a convex combination of the context values """

        output_values = torch.einsum("kij,ki->kj", node_values, attention_weights)
        
        return output_values

    def forward(self, context_embedding, node_embeddings, mask=None):

        context_query = self.context_query_projector(context_embedding)
        node_values = self.value_projector(node_embeddings)
        node_keys = self.key_projector(node_embeddings)
        
        compatibilities = self.compute_compatibilities(context_query, node_keys, mask)
        attention_weights = self.compute_attention_weights(compatibilities)
        context_embedding = self.compute_context_embedding(node_values, attention_weights)
        
        return context_embedding


class singleHeadedLogitDecoder(nn.Module):

    def __init__(self, input_size):
        super(singleHeadedLogitDecoder, self).__init__()
        
        self.input_size = input_size
    
        self.context_query_projector = nn.Linear(input_size, input_size, bias=False)
        self.value_projector = nn.Linear(input_size, input_size, bias=False)
        self.key_projector = nn.Linear(input_size, input_size, bias=False)
        
    def compute_compatibilities(self, context_query, node_keys, mask=None):
        """ Compute compatibilities of node keys with context query """
    
        compatibilities = torch.einsum("kj,kij->ki", context_query, node_keys)
        compatibilities = torch.clamp(compatibilities, -10, 10)
        compatibilities = compatibilities.tanh()
        compatibilities = compatibilities / (self.input_size) ** 0.5
        if mask is not None:
            compatibilities[mask] = - torch.inf

        return compatibilities

    def compute_attention_weights(self, compatibilities):
        """ Compute attention weights as a softmax over the compatibilities """

        attention_weights = torch.softmax(compatibilities, -1)
        
        return attention_weights
    
    def compute_context_embedding(self, node_values, attention_weights):
        """ Compute the context embedding as a convex combination of the node values """

        output_values = torch.einsum("kij,ki->kj", node_values, attention_weights)

        return output_values

    def forward(self, context_embedding, node_embeddings, mask=None):
        
        context_query = self.context_query_projector(context_embedding)
        node_values = self.value_projector(node_embeddings)
        node_keys = self.key_projector(node_embeddings)
        
        compatibilities = self.compute_compatibilities(context_query, node_keys, mask)
        attention_weights = self.compute_attention_weights(compatibilities)

        return attention_weights


class multiHeadedNodeEncoder(nn.Module):
    def __init__(self, input_size, query_size, key_size, value_size):
        super(multiHeadedNodeEncoder, self).__init__()

        self.multi_projectors = [
            self._set_object_attribute(
                nn.Linear(value_size, input_size, bias=False)
            )
            for num in range(int(input_size / query_size))
        ]
        self.single_headed_attention_encoders = [
            self._set_object_attribute(
                singleHeadedNodeEncoder(input_size, query_size, key_size, value_size)
            )
            for num in range(int(input_size / query_size))
        ]

    def _generate_random_hash(self):
        """ Generate a random hash using the current time """

        random_hash = hashlib.sha1()
        random_hash.update(str(time.time()).encode("utf-8"))
        random_hash = random_hash.hexdigest()[:8]
        
        return random_hash

    def _set_object_attribute(self, object):
        """ Set the given object as an attribute of the class """

        random_hash = self._generate_random_hash()
        self.__setattr__(random_hash, object)

        return object

    def forward(self, node_embeddings, mask=None):

        attention_embeddings = [
            attention_encoder(node_embeddings, mask)
            for attention_encoder in self.single_headed_attention_encoders
        ]
        attention_embeddings = [
            projector(attention_embedding)
            for (projector, attention_embedding) in zip(self.multi_projectors, attention_embeddings)
        ]

        attention_embeddings = torch.stack(attention_embeddings, axis=0)
        attention_embeddings = torch.sum(attention_embeddings, axis=0)

        return attention_embeddings



class multiHeadedContextNodeEncoder(nn.Module):
    def __init__(self, context_size, query_size, key_size, value_size):
        super(multiHeadedContextNodeEncoder, self).__init__()
                    
        self.multi_projectors = [
            self._set_object_attribute(
                nn.Linear(value_size, int(context_size / 3), bias=False)
            )
            for num in range(int(context_size / query_size / 3))
        ]
        self.single_headed_context_attention_encoders = [
            self._set_object_attribute(
                singleHeadedContextNodeEncoder(context_size, query_size, key_size, value_size)
            )
            for num in range(int(context_size / query_size / 3))
        ]

    def _generate_random_hash(self):
        """ Generate a random hash using the current time """
        
        random_hash = hashlib.sha1()
        random_hash.update(str(time.time()).encode("utf-8"))
        random_hash = random_hash.hexdigest()[:8]
        
        return random_hash
    
    def _set_object_attribute(self, object):
        """ Set the given object as an attribute of the class """

        random_hash = self._generate_random_hash()
        self.__setattr__(random_hash, object)
        
        return object

    def forward(self, context_embedding, node_embeddings, mask=None):

        attention_embeddings = [
            attention_encoder(context_embedding, node_embeddings, mask)
            for attention_encoder in self.single_headed_context_attention_encoders
        ]
        attention_embeddings = [
            projector(attention_embedding)
            for (projector, attention_embedding) in zip(self.multi_projectors, attention_embeddings)
        ]
        
        attention_embeddings = torch.stack(attention_embeddings, axis=0)
        attention_embeddings = torch.sum(attention_embeddings, axis=0)
        
        return attention_embeddings


class feedForwardNodeEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(feedForwardNodeEncoder, self).__init__()
        
        self.first_linear_encoder = nn.Linear(input_size, hidden_size)
        self.second_linear_encoder = nn.Linear(hidden_size, input_size)
        
    def forward(self, node_embeddings):
            
        node_embeddings = torch.unsqueeze(node_embeddings, 0)
        feedforward_embeddings = self.first_linear_encoder(node_embeddings)
        feedforward_embeddings = feedforward_embeddings.relu()
        feedforward_embeddings = self.second_linear_encoder(feedforward_embeddings)
        feedforward_embeddings = feedforward_embeddings.relu()
        feedforward_embeddings = torch.squeeze(node_embeddings, 0)
            
        return feedforward_embeddings


class transformerNodeEncoder(nn.Module):
    def __init__(self, input_size, query_size, key_size, value_size, hidden_size):
        super(transformerNodeEncoder, self).__init__()
        
        self.multi_headed_attention_encoder = multiHeadedNodeEncoder(
            input_size, query_size, key_size, value_size,
        )
        self.feedforward_encoder = feedForwardNodeEncoder(
            input_size, hidden_size,
        )
        self.layernorm = torch.nn.LayerNorm(input_size)
        
    def forward(self, node_embeddings):
            
        transformer_embeddings = self.multi_headed_attention_encoder(node_embeddings)
        transformer_embeddings += node_embeddings
        transformer_embeddings = self.layernorm(transformer_embeddings)
        transformer_embeddings += self.feedforward_encoder(transformer_embeddings)
        transformer_embeddings = self.layernorm(transformer_embeddings)
        
        return transformer_embeddings


class transformerEncoder(nn.Module):
    def __init__(self, input_size, query_size, key_size, value_size, hidden_size, num_layers):
        super(transformerEncoder, self).__init__()
        
        self.customer_linear_encoder = linearNodeEncoder(2, input_size)
        self.depot_linear_encoder = linearNodeEncoder(2, input_size)
        self.station_linear_encoder = linearNodeEncoder(3, input_size)
        self.transformer_node_encoders = [
            self._set_object_attribute(
                transformerNodeEncoder(input_size, query_size, key_size, value_size, hidden_size)
            )
            for num in range(num_layers)
        ]

    def _generate_random_hash(self):
        """ Generate a random hash using the current time """
        
        random_hash = hashlib.sha1()
        random_hash.update(str(time.time()).encode("utf-8"))
        random_hash = random_hash.hexdigest()[:8]
        
        return random_hash

    def _set_object_attribute(self, object):
        """ Set the given object as an attribute of the class """

        random_hash = self._generate_random_hash()
        self.__setattr__(random_hash, object)
        
        return object
        
    def _compute_depot_embedding(self, depot_features):

        if depot_features is not None:
            depot_embedding = self.depot_linear_encoder(depot_features)
        else:
            depot_embedding = None
            
        return depot_embedding

    def _compute_customer_embedding(self, customer_features):

        customer_embedding = self.customer_linear_encoder(customer_features)

        return customer_embedding

    def _compute_station_embedding(self, station_features=None):

        if station_features is not None:
            station_embedding = self.station_linear_encoder(station_features)
        else:
            station_embedding = None
            
        return station_embedding

    def _package_embeddings_concat(self, customer_embedding, depot_embedding=None, station_embedding=None):

        if depot_embedding is None:
            if station_embedding is None:
                transformer_embeddings = torch.concat([
                    customer_embedding
                ], -2)
            else:
                transformer_embeddings = torch.concat([
                    customer_embedding,
                    station_embedding
                ], -2)
        else:
            if station_embedding is None:
                transformer_embeddings = torch.concat([
                    depot_embedding,
                    customer_embedding
                ], -2)
            else:
                transformer_embeddings = torch.concat([
                    depot_embedding,
                    customer_embedding,
                    station_embedding
                ], -2)
            
        return transformer_embeddings

    def _package_embeddings_dict(self, customer_embedding, depot_embedding=None, station_embedding=None):

        transformer_embeddings = {
            "depot_embedding": depot_embedding,
            "customer_embedding": customer_embedding,
            "station_embedding": station_embedding,
        }
            
        return transformer_embeddings
    
    def forward(self, depot_features, customer_features, station_features=None, return_dict=False):

        depot_embedding = self._compute_depot_embedding(depot_features)
        customer_embedding = self._compute_customer_embedding(customer_features)
        station_embedding = self._compute_station_embedding(station_features)
        if not return_dict:
            transformer_embeddings = self._package_embeddings_concat(
                depot_embedding,
                customer_embedding,
                station_embedding
            )
        else:
            transformer_embeddings = self._package_embeddings_dict(
                depot_embedding,
                customer_embedding,
                station_embedding
            )
            
        return transformer_embeddings


class graphMeanEmbedder(nn.Module):
    def __init__(self):
        super(graphMeanEmbedder, self).__init__()
            
    def forward(self, transformer_embeddings):

        if transformer_embeddings.get("station_embedding") is not None:
            if transformer_embeddings.get("station_embedding") is not None:
                node_embeddings = torch.cat([
                    transformer_embeddings.get("depot_embedding"),
                    transformer_embeddings.get("customer_embedding"),
                    transformer_embeddings.get("station_embedding"),
                ], -2)
            else:
                node_embeddings = torch.cat([
                    transformer_embeddings.get("depot_embedding"),
                    transformer_embeddings.get("customer_embedding"),
                ], -2)
        else:
            if transformer_embeddings.get("station_embedding") is not None:
                node_embeddings = torch.cat([
                    transformer_embeddings.get("customer_embedding"),
                    transformer_embeddings.get("station_embedding"),
                ], -2)
            else:
                node_embeddings = torch.cat([
                    transformer_embeddings.get("customer_embedding"),
                ], -2)
        graph_embedding = torch.mean(node_embeddings, axis=1)            

        return graph_embedding       


class transformerEmbedder(nn.Module):
    def __init__(self, input_size, query_size, key_size, value_size, hidden_size, num_layers):
        super(transformerEmbedder, self).__init__()
        
        self._transformer_encoder = transformerEncoder(
            input_size, query_size, key_size, value_size, hidden_size, num_layers)
    
    def forward(self, depot_features, customer_features, station_features=None, return_dict=False):

        transformer_embeddings = self._transformer_encoder(
            depot_features=depot_features,
            customer_features=customer_features,
            station_features=station_features,
            return_dict=return_dict,
        )
        graph_embedding = torch.mean(transformer_embeddings, axis=1)
        
        return transformer_embeddings, graph_embedding


