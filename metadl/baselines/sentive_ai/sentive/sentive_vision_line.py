class sentive_vision_line(object):
    """
        prototype de données des objets neurones sentive vision curve
        Cette classe ne contient aucune fonction mais sert à conserver les données
    """ 
    def __init__(self, number):
        self.number = number
        self.neuron = {
            "_id": number,
            "schema_version":1,
            "type": "sentive_vision_line",
            "layer_id":0,
            "DbConnectivity":{
                "pre_synaptique":[],
                "post_synaptique":[],
                "lateral_connexion":[],
                "weights":{}
            },
            "meta":{
                "threshold" : 0.7,
                "pending_nb_iteration" : 0,
                "averaged_prediction" : 0.0,
                "nb_points_aligned" : 1,
                "cumulated_angle" : 0.0,
                "line" : {
                    "starting_point" : {
                        "x" : 0.0,
                        "y" : 0.0
                    },
                    "basis_vector" : {
                        "x" : 0.0,
                        "y" : 0.0
                    },
                    "last_position" : {
                        "x" : 0.0,
                        "y" : 0.0
                    },
                    "nb_iteration" : 0
                },
                "pixels_matrix": []
            }
        }