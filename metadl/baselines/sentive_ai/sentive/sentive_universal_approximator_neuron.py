class sentive_universal_approximator_neuron(object):
    """prototype de neurone permettant d'approximer n'importe quelle fonction par des segments de droites affines

    Olivier Manette
    10 novembre 2021

    Args:
        object (int): identifiant
    """
    def __init__(self, number):
        self.number = number
        self.neuron = {
            "_id" : number, 
            "schema_version" : 1,
            "type" : "sentive_universal_approximator_neuron",
            "layer_id" : 0,
            "DbConnectivity":{
                "pre_synaptique" : [],
                "post_synaptique" : [],
                "lateral_connexion" : [],
                "weights":{}
            },
            "meta":{
                "confidence_index" : 0
            }
        }