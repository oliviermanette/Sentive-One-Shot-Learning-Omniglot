class sentive_arc_neuron(object):
    """
        prototype de données des objets neurones vision segment arcs
        Cette classe ne contient aucune fonction mais ne sert qu'à conserver les données
        Les fonctions sont dans une autre classe.
    """ 
    def __init__(self, number):
        self.number = number
        self.neuron = {
            "_id": number,
            "schema_version":1,
            "type": "sentive_vision_arcs",
            "layer_id":0,
            "ratio_conn":0,
            "DbConnectivity":{
                "pre_synaptique":[],
                "post_synaptique":[],
                "lateral_connexion":[],
                "weights":{}
            },
            "meta":{
                "last_nrn2" : {
                    'glbl_prm': {
                        'cg': {
                            'x': 0.0, 
                            'y': 0.0
                        },
                        'u_axis': {
                            'x': 0.0, 
                            'y': 0.0
                        }
                    },
                    "vecteur_deplacement": {
                            'x': 0.0, 
                            'y': 0.0
                    }
                },
                "threshold": 0.7,
                "default_malm": 1.0,
                "curve" : {
                    "starting_point" : {
                        "x" : 0.0,
                        "y" : 0.0
                    },
                    "basis_vector" : {
                        "x" : 0.0,
                        "y" : 0.0
                    },
                    "nb_iteration" : 0,
                    "rotation_angle" : 0.0,
                    "angle_u_axis" : 0.0,
                    "angle_v_deplacement" : 0.0,
                    "malm_rotation_angle" : 0.0,
                    "acceleration_step" : 0.0,
                    "distance" : 0.0
                }
            }
        }