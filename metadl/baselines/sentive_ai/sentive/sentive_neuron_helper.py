import numpy as np
import matplotlib.pyplot as plt
import copy

import networkx as nx

from .sentive_vision_neuron import sentive_vision_neuron
from .sentive_sequence_nrn import sentive_sequence_nrn
from .sentive_angle_neuron import sentive_angle_neuron
from .sentive_arc_neuron import sentive_arc_neuron
from .sentive_vision_curve import sentive_vision_curve
from .sentive_vision_line import sentive_vision_line

class sentive_neuron_helper():
    def __init__(self):
        
        self.init_matrix = []
        self.init_matrix.append( np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]]))
        self.init_matrix.append( np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]]))
        self.init_matrix.append( np.array([[1, 1, 0],[1, 0, -1],[0, -1, -1]]))
        self.init_matrix.append( np.array([[0, -1, -1],[1, 0, -1],[1, 1, 0]]))


        self.dir_matrix =  np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]])


        self.ok_conf = []
        self.ok_conf.append( np.array([[1, 0, -1],[0, 1, 0],[0, 0, 0]]))
        self.ok_conf.append( np.array([[1, 0, 0],[0, 1, -1],[0, 0, 0]]))
        self.ok_conf.append( np.array([[1, 0, 0],[0, 1, 0],[0, 0, -1]]))
        
        self.ok_conf.append( np.array([[0, 0, -1],[1, 1, 0],[0, 0, 0]]))
        self.ok_conf.append( np.array([[0, 0, 0],[1, 1, -1],[0, 0, 0]]))
        self.ok_conf.append( np.array([[0, 0, 0],[1, 1, 0],[0, 0, -1]]))
        
        self.ok_conf.append( np.array([[0, 0, -1],[0, 1, 0],[1, 0, 0]]))
        self.ok_conf.append( np.array([[0, 0, 0],[0, 1, -1],[1, 0, 0]]))
        self.ok_conf.append( np.array([[0, 0, 0],[0, 1, 0],[1, 0, -1]]))
        
        self.ok_conf.append( np.array([[0, 0, 1],[0, 1, 0],[0, 0, -1]]))
        self.ok_conf.append( np.array([[0, 1, 0],[0, 1, 0],[0, 0, -1]]))
        
        self.ok_conf.append( np.array([[1, 0, 0],[0, 1, 0],[0, -1, 0]]))
        self.ok_conf.append( np.array([[0, 1, 0],[0, 1, 0],[0, -1, 0]]))
        
        self.ok_conf.append( np.array([[1, 0, 0],[0, 1, 0],[-1, 0, 0]]))
        self.ok_conf.append( np.array([[0, 1, 0],[0, 1, 0],[-1, 0, 0]]))
        
        self.lst_nrns = []
        self.id_nrn = 0 # id max des neurones
        self.nb_nrns = 0 # nb de neurones dans le tableau (sans avoir à utiliser la fonction len)
        
        # https://networkx.org/documentation/stable/tutorial.html
        self.netGraph = nx.Graph()

        self.layer_nb = 0
        self.layer_graph = []

        self.nb_2_1st_layers = 0
        self.pos_nrn_by_layer = []


    def new_layer(self):
        self.layer_nb +=1
        self.layer_graph.append(nx.DiGraph())
        self.pos_nrn_by_layer.append(self.nb_nrns)

    
    def add_edge(self, nrn1_id, nrn2_id):
        self.netGraph.add_edge(nrn1_id, nrn2_id)
        if self.layer_nb>0:
            self.layer_graph[self.layer_nb-1].add_edge(nrn1_id, nrn2_id)


    def increment_weight(self, nrn, nrn_post_synaptic_id):
        try:
            nrn["DbConnectivity"]["weights"][nrn_post_synaptic_id] += 1
        except KeyError:
            nrn["DbConnectivity"]["weights"][nrn_post_synaptic_id] = 1

        
    def add_nrn_connexion(self, nrn_post, nrn_pre_synaptic):
        nrn_post["DbConnectivity"]["pre_synaptique"].append(nrn_pre_synaptic["_id"])
        nrn_pre_synaptic["DbConnectivity"]["post_synaptique"].append(nrn_post["_id"])
        self.increment_weight(nrn_pre_synaptic, nrn_post["_id"])
        self.add_edge(nrn_pre_synaptic["_id"], nrn_post["_id"])


    def add_nrn_lateral(self, nrn_pre_synaptic, nrn_post_id ):
        # check si la connexion existe déjà
        for nrn_id in nrn_pre_synaptic["DbConnectivity"]["lateral_connexion"]:
            if nrn_id == nrn_post_id:
                self.increment_weight(nrn_pre_synaptic, nrn_post_id)
                return
        nrn_pre_synaptic["DbConnectivity"]["lateral_connexion"].append(nrn_post_id)
        self.increment_weight(nrn_pre_synaptic, nrn_post_id)
        self.add_edge(nrn_pre_synaptic["_id"], nrn_post_id)
        


    def add_new_nrn(self, nrn_type=''):
        """Ajoute un nouveau neurone au pool (remplace la base de données MongoDB de Sentive AI en mode non cloud)

        Returns:
            [int]: [identifiant du nouveau neurone créé]
        """
        self.id_nrn += 1
        if nrn_type=='':
            self.lst_nrns.append(sentive_vision_neuron(self.id_nrn))
        elif nrn_type=="sentive_sequence_nrn":
            self.lst_nrns.append(sentive_sequence_nrn(self.id_nrn))
        elif nrn_type=="sentive_angle_neuron":
            self.lst_nrns.append(sentive_angle_neuron(self.id_nrn))
        elif nrn_type=="sentive_arc_neuron":
            self.lst_nrns.append(sentive_arc_neuron(self.id_nrn))
        elif nrn_type=="sentive_vision_curve":
            self.lst_nrns.append(sentive_vision_curve(self.id_nrn))
        elif nrn_type=="sentive_vision_line":
            self.lst_nrns.append(sentive_vision_line(self.id_nrn))
        
        self.netGraph.add_node(self.id_nrn)

        if self.layer_nb>0:
            self.layer_graph[self.layer_nb-1].add_node(self.id_nrn)

        self.nb_nrns = len(self.lst_nrns)
        self.lst_nrns[self.nb_nrns-1].neuron["layer_id"] = self.layer_nb

        return self.nb_nrns - 1
        
        
    def FctIterMean(self, Nb_activations, NewAct, avgValue):
        """Calcule la Moyenne itérative

        Args:
            Nb_activations ([int]): [nb de valeur intégrée dans la moyenne précédente]
            NewAct ([float]): [Nouvelle valeur à intégrer à la moyenne]
            avgValue ([float]): [valeur moyenne précédemment calculée]

        Returns:
            [float]: [nouvelle moyenne]
        """
        Nb_activations = int(Nb_activations)
        NewAct = float(NewAct)
        avgValue = float(avgValue)
        return ((Nb_activations - 1) / Nb_activations
                * avgValue + NewAct / Nb_activations)
    
    
    def get_x_matrix(self, size):
        size = int(size)
        if size>=2:
            output = np.array([np.arange(size),np.arange(size)])
        else:
            return np.array(np.arange(size))
        for i in range(2,size):
            output = np.append(output,[np.arange(size)],axis=0)
        return output

    
    def get_y_matrix(self, size):
        size = int(size)
        if size>=2:
            output = np.array([np.ones(size)*0,np.ones(size)*1])
        else:
            return np.array(np.arange(size))
        for i in range(2,size):
            output = np.append(output,[np.ones(size)*i],axis=0)
        return output
    
    
    def get_matrix_center(self, size):
        """Retourne les coordonnées du centre de la matrice de taille "size"

        Args:
            size ([int]): [de prédérence une matrice carré de taille impaire]

        Returns:
            [int]: [coordonnées x et y du centre de la matrice carré impaire]
        """
        return np.floor(size/2)
    
    
    def get_receptive_field(self, local_neuron, current_vision):
        """
            
        """
        min_val_y = int(local_neuron["meta"]["center"]["y"] - np.floor(
                                                local_neuron["meta"]["matrix_width"]/2))
        max_val_y = int(local_neuron["meta"]["center"]["y"] + np.ceil(
                                                local_neuron["meta"]["matrix_width"]/2))
        min_val_x = int(local_neuron["meta"]["center"]["x"] - np.floor(
                                                local_neuron["meta"]["matrix_width"]/2))
        max_val_x = int(local_neuron["meta"]["center"]["x"] + np.ceil(
                                                local_neuron["meta"]["matrix_width"]/2))
        return current_vision[min_val_y:max_val_y, min_val_x:max_val_x, 0]
    

    def get_all_center_fields(self, list_neurons, current_vision):
        """
            Retourne l'image avec les centres des neurones surlignés
            Pour l'ensemble des neurones
        """
        nb = 0
        for sent_neuron in list_neurons:
            neuron = sent_neuron.neuron["meta"]
            current_vision[neuron["center"]["y"],neuron["center"]["x"]] = 5 #* current_vision[neuron["center"]["y"],neuron["center"]["x"]]
            nb += 1
        print(nb,"neurons")
        return current_vision
    
    
    def get_all_center_fields_width(self, list_neurons, current_vision, lint_width=5):
        """
            Retourne l'image avec les centres des neurones surlignés
            Il faut spécifier la couche des neurones sélectionnés
        """
        nb = 0
        for sent_neuron in list_neurons:
            neuron = sent_neuron.neuron["meta"]
            if neuron["matrix_width"] == lint_width:
                current_vision[neuron["center"]["y"],neuron["center"]["x"]] = 5 #* current_vision[neuron["center"]["y"],neuron["center"]["x"]]
                nb += 1
        print(nb,"neurons")
        return current_vision

    
    def get_neuron_receptive_field(self, nrn_id, current_vision, neurons_pool=-1, verbose=False):
        """Retourne le champs récepteur du neurone sur la matrice current_vision.

        Args:
            current_vision ([type]): [description]
            nrn_id ([type]): [description]
            neurons_pool (int, optional): [description]. Defaults to -1.
            verbose (bool, optional): [description]. Defaults to False.

        Returns:
            [matrice]: [matrice contenant la position du champs récepteur du neurone nrn_id]
        """
        tmp_vision = copy.deepcopy(current_vision)


        if neurons_pool==-1:
            neurons_pool = self.lst_nrns
            
        # récupère le neurone visé
        crnt_nrn = self.get_neuron_from_id(nrn_id, neurons_pool)

        # si le neurone est un neurone pixel alors tu remplaces la valeur de la matrice par son weight
        if crnt_nrn["layer_id"] == 1:
            pixel_value = crnt_nrn["weight"]
        else:
            pixel_value = 5

        if crnt_nrn["layer_id"] == 1:
            tmp_vision[int(crnt_nrn["meta"]["center"]["y"]),int(crnt_nrn["meta"]["center"]["x"])] = pixel_value
            return tmp_vision
        # récupère la liste des 
        try:
            lst_nrn = crnt_nrn["meta"]["field_list"]
        except KeyError:
            lst_nrn = crnt_nrn["DbConnectivity"]["pre_synaptique"]
        except TypeError:
            print("nrn_id",nrn_id, "crnt_nrn", crnt_nrn)
        
        # récupère le neurone pour chaque id de la liste
        nb = 0
        for sensor_id in lst_nrn:
            neuron = self.get_neuron_from_id(sensor_id, neurons_pool)
            if neuron !="":
                tmp_vision[int(neuron["meta"]["center"]["y"]),int(neuron["meta"]["center"]["x"])] = pixel_value
                nb +=1
        if verbose:
            print(nb, "pixels")
            print(crnt_nrn)
        return tmp_vision
    
    
    def update_coord(self, previous):
        """
            lorsqu'on augmente la taille de la matrice de +2
            Toutes les coordonnées relatives à la taille précédente doivent être modifiées grace 
            à cette fonction.
        """
        previous["x"] += 1
        previous["y"] += 1
        return previous
    

    def get_list_pixels_coord(self, nrn_id):
        """ Returns the list of pixels coordinates of the receptive field of the
        neuron with id nrn_id.
        """
        lst_pixels_coord = []
        nrn_source = self.get_neuron_from_id(nrn_id)
        if nrn_source is None:
            return None

        for nrn_id in nrn_source["DbConnectivity"]["pre_synaptique"]:
            nrn_pixel = self.get_neuron_from_id(nrn_id)
            lst_pixels_coord.append(nrn_pixel["meta"]["center"])
        return lst_pixels_coord
    
    
    def rotate_vector(self, vector, angle_rotation):
        """Retourne les coordonnées du vector après rotation
        TODO: cette fonction n'est semble t'il jamais appelée

        Args:
            vector ([struct]): [structure contenant les coordonnées (x,y) d'un vecteur]
            angle_rotation ([float]): [exprimé en radian]

        Returns:
            [type]: [description]
        """
        output_vector = {
            "x":0,
            "y":0
        }
        output_vector["x"] = np.around(vector["x"] * np.cos(angle_rotation) - vector["y"] * np.sin(angle_rotation))
        output_vector["y"] = np.around(vector["x"] * np.sin(angle_rotation) + vector["y"] * np.cos(angle_rotation))
        return output_vector
    
    
    def anti_rotate_vector(self, vector, angle_rotation):
        output_vector = {
            "x":0,
            "y":0
        }
        output_vector["x"] = np.around(vector["x"] * np.cos(angle_rotation) + vector["y"] * np.sin(angle_rotation))
        output_vector["y"] = np.around(vector["y"] * np.cos(angle_rotation) - vector["x"] * np.sin(angle_rotation))
        return output_vector
    
    
    def get_pos_from_id(self, neuron_idx2, neurons_pool=-1):
        """
            retourne la position dans la tableau à partir du neuron_id
        """
        if neurons_pool==-1:
            neurons_pool = self.lst_nrns
        for neuron_idx in range(len(neurons_pool)):
            if neurons_pool[neuron_idx].neuron["_id"]==neuron_idx2:
                break
        return neuron_idx
    
    
    def get_neuron_from_id(self, neuron_idx2, neurons_pool=-1, str_id="_id"):
        """
            retourne le neurone à partir de son neuron_id "_id"
        """
        if neurons_pool==-1:
            neurons_pool = self.lst_nrns
        for neuron_idx in range(len(neurons_pool)):
            if neurons_pool[neuron_idx].neuron[str_id]==neuron_idx2:
                return neurons_pool[neuron_idx].neuron
        return {"_id":None}


    def get_segment_from_id(self, neuron_idx2, neurons_pool=-1, str_id="_id"):
        """
            retourne le neurone à partir de son neuron_id "_id"
        """
        if neurons_pool==-1:
            neurons_pool = self.lst_nrns
        for neuron_idx in range(len(neurons_pool)):
            if neurons_pool[neuron_idx][str_id]==neuron_idx2:
                return neurons_pool[neuron_idx]
        return ''
    
    
    def get_avg_center(self, list_neuron_ids, neurons_pool=-1):
        """
            retourne la moyenne des centres à partir des neurones_id passés en paramètres
        """
        if neurons_pool==-1:
            neurons_pool = self.lst_nrns
            
        list_x = []
        list_y = []

        output={
            "center":{
                "x":0,
                "y":0
            },
            "real_center":{
                "x":0,
                "y":0
            }
        }

        for int_id in list_neuron_ids:
            current_neuron = self.get_neuron_from_id(int_id, neurons_pool)
            list_x.append(current_neuron["meta"]["center"]["x"])
            list_y.append(current_neuron["meta"]["center"]["y"])

        output["real_center"]["y"]=np.mean(list_y)
        output["real_center"]["x"]=np.mean(list_x)

        output["center"]["x"]= int(np.round(output["real_center"]["x"]))
        output["center"]["y"] = int(np.round(output["real_center"]["y"]))
        return output

    
    def calc_angle(self, vector1, vector2):
        # calcul de l'angle de rotation entre les deux vecteurs passés en paramètres
        np_c_1 = np.array([vector1["x"], vector1["y"]])
        # print (np_c_1)
        np_c_2 = np.array([vector2["x"], vector2["y"]])
        # print (np_c_2)
        np_c_3 = np.array([-vector1["y"], vector1["x"]])
        signe = 1
        test = np.sum(np.multiply(np_c_3,np_c_2))
        if test < 0 :
            signe = -1
        tmp_value = np.sum(np.multiply(np_c_1,np_c_2))/(np.sqrt(np.sum(np.power(np_c_1,2)))*np.sqrt(np.sum(np.power(np_c_2,2))))
        if tmp_value<-1:
            return signe * np.pi
        elif tmp_value > 1:
            return 0
        angle =  signe * np.arccos(tmp_value)
        if np.isnan(angle):
            return signe * np.arccos(np.round(np.sum(np.multiply(np_c_1,np_c_2))/(np.sqrt(np.sum(np.power(np_c_1,2)))*np.sqrt(np.sum(np.power(np_c_2,2))))))
        return angle
    
    
    def calc_dist(self, point1, point2):
        """Calcule la distance entre deux points

        Args:
            point1 ([struct]): [description]
            point2 ([struct]): [description]

        Returns:
            [float]: [distance exprimé dans la même unités que les coordonnées des points passés en paramètres]
        """
        X_D = pow(point1["x"] - point2["x"],2)
        Y_D = pow(point1["y"] - point2["y"],2)
        return pow(X_D+Y_D,0.5)


    def calc_total_distance(self, nrn_list, neurons_pool=-1):
        output_total = 0
        if len(nrn_list)==0: return 0
        if neurons_pool==-1:
            neurons_pool = self.lst_nrns
        nrn = self.get_neuron_from_id(nrn_list[0], neurons_pool)
        try:
            point1 = nrn["meta"]["center"]
        except:
            return 0
        for nrn_pos in range(1, len(nrn_list)):
            try:
                nrn_id = nrn_list[nrn_pos]
                nrn = self.get_neuron_from_id(nrn_id, neurons_pool)
                point2 = nrn["meta"]["center"]
                sub_dist = self.calc_dist(point1, point2)
                output_total += sub_dist
                point1 = point2
            except:
                pass
        return output_total


    def get_gbl_orientO(self, nrn):
        hand_1 = nrn["meta"]["local_tip_1"]
        hand_2 = nrn["meta"]["local_tip_2"]
        v_outpt = {"x":0,"y":0}
        
        if hand_1["x"]< hand_2["x"]:
            v_outpt["x"] = hand_2["x"] - hand_1["x"]
            v_outpt["y"] = hand_2["y"] - hand_1["y"]
        elif hand_1["x"] > hand_2["x"]:
            v_outpt["x"] = hand_1["x"] - hand_2["x"]
            v_outpt["y"] = hand_1["y"] - hand_2["y"]
        elif hand_1["y"]< hand_2["y"]:
            v_outpt["x"] = hand_2["x"] - hand_1["x"]
            v_outpt["y"] = hand_2["y"] - hand_1["y"]
        elif hand_1["y"] > hand_2["y"]:
            v_outpt["x"] = hand_1["x"] - hand_2["x"]
            v_outpt["y"] = hand_1["y"] - hand_2["y"]
        return v_outpt

    
    def get_global_orientation(self, nrn_id, neurons_pool=-1):
        """Retourne le vecteur allant directement d'une extrémité à l'autre
        du champs récepteur du neurone
        Globalement orienté de gauche à droite et sinon de bas en haut.

        Args:
            nrn_id (int): identifiant du neurone
            neurons_pool (list, optional): base de données des neurones. Defaults to -1.

        Returns:
            struct: vecteyr donnant l'orientation générale
        """
        if neurons_pool==-1:
            neurons_pool = self.lst_nrns

        nrn = self.get_neuron_from_id(nrn_id, neurons_pool)

        return self.get_gbl_orientO(nrn)


    def check_curve_geometry(self, starting_point, basis_vector, starting_angle, acceleration, nb_min_iterations, lst_checkpoints):
        # on initialise les distance à partir du starting_point
        list_distances = []
        for checkpoint in lst_checkpoints:
            list_distances.append(self.calc_dist(starting_point, checkpoint))

        # on effectue une première translation du vecteur de départ
        current_position = {
            "x": starting_point["x"] + basis_vector["x"],
            "y": starting_point["y"] + basis_vector["y"]
        }
        # calcul des distances depuis la nouvelle position mais avec un itérateur donnant l'indice de la liste
        self.get_closest_dist_for_checkpoints(lst_checkpoints, list_distances, current_position)

        j = 0
        # on commence par une boucle for correspondant au nombre d'itérations minimum
        for i in range(nb_min_iterations - 1):
            # on effectue une rotation du vecteur
            basis_vector = self.draw_rotate_vector(basis_vector, starting_angle)
            # on effectue une translation du vecteur
            current_position["x"] += basis_vector["x"]
            current_position["y"] += basis_vector["y"]
            # calcul des distances depuis la nouvelle position mais avec un itérateur donnant l'indice de la liste
            self.get_closest_dist_for_checkpoints(lst_checkpoints, list_distances, current_position)
            # modification de l'angle de rotation par l'accélération
            starting_angle += acceleration
            j+=1

        nb_elements = len(lst_checkpoints)
        min_distance = copy.deepcopy(list_distances[nb_elements-1])
        distance = copy.deepcopy(list_distances[nb_elements-1])
        
        # on continue dans une boucle while jusqu'à ce que la dernière distance soit minimale
        while list_distances[nb_elements-1] == min_distance:
            # on effectue une rotation du vecteur
            basis_vector = self.draw_rotate_vector(basis_vector, starting_angle)
            # on effectue une translation du vecteur
            current_position["x"] += basis_vector["x"]
            current_position["y"] += basis_vector["y"]
            # calcul des distances depuis la nouvelle position mais avec un itérateur donnant l'indice de la liste
            distance = copy.deepcopy(self.calc_dist(current_position, lst_checkpoints[nb_elements-1]))
            j += 1
            if distance < min_distance:
                min_distance = copy.deepcopy(distance)
                list_distances[nb_elements-1] = copy.deepcopy(distance)
            if distance < 1:
                break
            if j > 100:
                break

            # modification de l'angle de rotation par l'accélération
            starting_angle += acceleration

        # print("°-° : distance nb itérations:", j)
        return list_distances


    def get_closest_dist_for_checkpoints(self, lst_checkpoints, list_distances, current_position):
        for i, checkpoint in enumerate(lst_checkpoints):
            new_dist = self.calc_dist(current_position, checkpoint)
            if new_dist < list_distances[i]:
                list_distances[i] = new_dist

    
    def draw_rotate_vector(self, vector, angle_rotation):
        """
        Retourne un angle après rotation
        Ne fait pas d'arrondi contrairement à l'autre fonction rotate_vector
        """
        output_vector = {
            "x":0,
            "y":0
        }
        output_vector["x"] = vector["x"] * np.cos(angle_rotation) - vector["y"] * np.sin(angle_rotation)
        output_vector["y"] = vector["x"] * np.sin(angle_rotation) + vector["y"] * np.cos(angle_rotation)
        return output_vector
    

    def draw_line(self, mtrx, init_pos, new_pos):
        # Allumer toutes les positions intermédiaires
        # Compter le nombre de pixels intermédiaires
        # En X et en Y
        nb_x = new_pos["x"] - init_pos["x"]
        nb_y = new_pos["y"] - init_pos["y"]

        # Sauvegarder dans un tableau la position initiale
        save_pos = [(init_pos["x"], init_pos["y"])]
        # Sauvegarder dans un tableau la position finale
        save_pos.append((new_pos["x"], new_pos["y"]))

        if np.abs(nb_x) > np.abs(nb_y):
            if nb_x < 0:
                signe = -1
            else:
                signe = 1
            for i in range(1, np.abs(nb_x)):
                proportion = i/nb_x
                crt_pos_x = init_pos["x"] + (i * signe)
                crt_pos_y = int(round(init_pos["y"] + signe * proportion * nb_y))
                # sauvegarder la position
                save_pos.append((crt_pos_x, crt_pos_y))
                try:
                    mtrx[crt_pos_y][crt_pos_x] = 1
                except:
                    pass
        else:
            if nb_y > 0:
                signe = 1
            else:
                signe = -1
            for i in range(1, np.abs(nb_y)):
                proportion = i/nb_y
                crt_pos_y = init_pos["y"] + signe * i
                crt_pos_x = int(round(init_pos["x"] + signe * proportion * nb_x))
                # sauvegarder la position
                save_pos.append((crt_pos_x, crt_pos_y))
                try:
                    mtrx[crt_pos_y][crt_pos_x] = 1
                except:
                    pass
        return mtrx, save_pos
    
    def intensite_pixel_local(self, x):
        if x >= 0 and x <= 0.5:
            y = x + 0.5
        elif x > 0.5 and x <= 1:
            y = -x + 1.5
        else:
            print("La valeur de x doit être comprise entre 0 et 1.")
            return 0
        return y
    

    def nrn_drawer(self, mtrx, vector, angle, length, start, pente = 0):
        """
        Dessine un segment de courbe
        ============================
        En plus de la matrice dans laquelle il va dessiner, il ne prend que 4 paramètres.
        Le vecteur de départ, angle de rotation, la longueur (ou le nombre d'itérations).
        Et le point de départ.
        """
        # initialisation de la matrice
        mtrx[start["y"]][start["x"]] = 1
        print("**nrn_drawer** : position de départ x", start["x"],"y",start["y"])

        # si la position est un entier, on ajoute 0.5 pour ramener au centre du pixel
        if np.floor(start["x"])==start["x"] and np.floor(start["y"])==start["y"]:
            new_pos = {"x": start["x"] + 0.5, "y": start["y"] + 0.5}
            tmp_pos = {"x": start["x"] + 0.5, "y": start["y"] + 0.5}
        else:
            # sinon on la garde telle quelle
            new_pos = {"x": start["x"], "y": start["y"]}
            tmp_pos = {"x": start["x"], "y": start["y"]}

        # on décale le position de départ avec le premier vecteur
        tmp_pos["x"] = tmp_pos["x"]+vector["x"]
        new_pos["x"] = int(np.floor(tmp_pos["x"]))
        tmp_pos["y"] = tmp_pos["y"]+vector["y"]
        new_pos["y"] = int(np.floor(tmp_pos["y"]))
        
        # assigner les intensités des pixels
        mtrx = self.assign_pixels_intensities(mtrx, new_pos, tmp_pos)

        # print("position premier vecteur x",new_pos["x"],"y", new_pos["y"])

        _, save_points =  self.draw_line(mtrx, start, new_pos)

        for i in range(length-1):
            # Sauvegarde de la position initiale
            init_pos = copy.deepcopy(new_pos)

            # rotate vector : position finale
            # calcul de la nouvelle position
            # print("\n*******************")
            # print("angle", angle, "pente", pente, "i", i)
            vector = self.draw_rotate_vector(vector, angle)
            if pente !=0:
                angle += pente

            # print("vector intermédiaire", vector)
            # print("i", i)
            tmp_pos["x"] = tmp_pos["x"] + vector["x"]
            new_pos["x"] = int(np.floor(tmp_pos["x"]))
            tmp_pos["y"] = tmp_pos["y"]+vector["y"]
            new_pos["y"] = int(np.floor(tmp_pos["y"]))
            print("**nrn_drawer** : position intermédiaire x",new_pos["x"], "y", new_pos["y"])
            
            # assigner les intensités des pixels
            mtrx = self.assign_pixels_intensities(mtrx, new_pos, tmp_pos)

            _, tmp_points = self.draw_line(mtrx, init_pos, new_pos)
            save_points.extend(tmp_points)
        
        print("**nrn_drawer** : angle final:", angle )
        print("**nrn_drawer** : position finale x",new_pos["x"],"y", new_pos["y"])
        print("**nrn_drawer** : dernier vector", vector)
        # supprimer les doublons de save_points
        save_points = list(set(save_points))
        return mtrx, angle, new_pos, vector, save_points


    def assign_pixels_intensities(self, mtrx, new_pos, tmp_pos):
        # Vérifie que les positions new_pos sont à l'intérieur de la mtrx

        # intensité du pixel local est toujours 1
        try:
            mtrx[new_pos["y"]][new_pos["x"]] = 1
        except:
            return mtrx
        # calcul de l'intensité des pixels adjacents
        # l'intensité est calculée par la fonction y = 2x-1
        # la position dans le pixel est la valeur x et la valeur y est l'intensité des pixels adjacents
        # Si y est négatif ça correspond à l'intensité du pixel d'avant. Le pixel d'après restant noir.
        # Si y est positif c'est le pixel d'après qui prend la valeur de y tandis que celui d'avant reste à zéro.

        # calcul de l'intensité des pixels adjacents sur l'axe des x
        pos_x = np.floor(tmp_pos["x"])
        reste_x = tmp_pos["x"] - pos_x
        # calcul de l'intensité
        x_intensite = 2 * reste_x - 1
        if x_intensite < 0:
            # si l'intensité est négative, on met le pixel d'avant à abs(x_intensite)
            try:
                # si l'intensité du pixel est zéro, on met la valeur absolue de x_intensite
                if mtrx[new_pos["y"]][new_pos["x"]-1] == 0:
                    mtrx[new_pos["y"]][new_pos["x"]-1] = np.abs(x_intensite)
            except:
                pass
        else:
            # sinon on met le pixel d'après à x_intensite
            try:
                # si l'intensité du pixel est zéro, on met la valeur absolue de x_intensite
                if mtrx[new_pos["y"]][new_pos["x"]+1] == 0:
                    mtrx[new_pos["y"]][new_pos["x"]+1] = x_intensite
            except:
                pass

        # calcul de l'intensité des pixels adjacents sur l'axe des y
        pos_y = np.floor(tmp_pos["y"])
        reste_y = tmp_pos["y"] - pos_y
        # calcul de l'intensité
        y_intensite = 2 * reste_y - 1
        if y_intensite < 0:
            # si l'intensité est négative, on met le pixel d'avant à abs(x_intensite)
            try:
                # si l'intensité du pixel est zéro, on met la valeur absolue de x_intensite
                if mtrx[new_pos["y"]-1][new_pos["x"]] == 0:
                    mtrx[new_pos["y"]-1][new_pos["x"]] = np.abs(y_intensite)
            except:
                pass
        else:
            # sinon on met le pixel d'après à x_intensite
            try:
                # si l'intensité du pixel est zéro, on met la valeur absolue de x_intensite
                if mtrx[new_pos["y"]+1][new_pos["x"]] == 0:
                    mtrx[new_pos["y"]+1][new_pos["x"]] = y_intensite
            except:
                pass
        return mtrx
    

    def char_drawer(self, mtrx, start_point, tbl_code_char):
        """Dessine un caractère complet.
        mtrx: matrice dans laquelle on dessine
        start_point: struct, "x", "y"; coordonnées de la position de départ
        tbl_code_char: tableau 3xN contenant les instructions de dessin
        """
        pass


    def get_list_presyn(self, lst_nrn, neurons_pool=-1):
        """retourne la liste des neurones pre_synaptique à partir d'une liste d'Identifiant et 

        Args:
            lst_nrn ([list de integer]): [id des neurones]
            neurons_pool ([list de sentive_vision_neurons]): [base de données des neurones dans laquelle chercher]

        Returns:
            [list d'integer]: [les id des neurones présynaptique pour tous les neurones passés en entrée]
        """
        if neurons_pool==-1:
            neurons_pool = self.lst_nrns
        lst_output = []
        for nrn_id in lst_nrn:
            lst_output.extend(self.get_neuron_from_id(nrn_id, neurons_pool)["DbConnectivity"]["pre_synaptique"])
        # lst_output = list(set(lst_output.sort()))
        return lst_output
    

    def intersect_presyn_field_list(self, nrn_id_1, nrn_id_2, neurons_pool=-1):
        if neurons_pool==-1:
            neurons_pool = self.lst_nrns
        lst_nrn_1 = self.get_neuron_from_id(nrn_id_1, neurons_pool)["meta"]["field_list"]
        # print("lst_nrn_1",lst_nrn_1)
        list1 = self.get_list_presyn(lst_nrn_1, neurons_pool)
        lst_nrn_2 = self.get_neuron_from_id(nrn_id_2, neurons_pool)["meta"]["field_list"]
        list2 = self.get_list_presyn(lst_nrn_2, neurons_pool)
        return list(set(list1).intersection(list2))


    def calc_tips(self, neuron_id, neurons_pool=-1):
        """A partir de real_center calcule les distances avec chaque point de field list
        sélectionne les 2 neurones les plus éloignés du centre.
        Ce sont a priori les extrémités du segment.

        Args:
            neuron ([sentive_vision_neuron]): [description]

        Returns:
            [sentive_vision_neuron]: [modifié avec les bonnes informations des tips]
        """
        output = {
            "local_tip_1":{
                "x":0,
                "y":0
            },
            "vector_1":{
                "x":0,
                "y":0
            },
            "local_tip_2":{
                "x":0,
                "y":0
            },
            "vector_2":{
                "x":0,
                "y":0
            },
            "length_c":0
        }
        if neurons_pool==-1:
            neurons_pool = self.lst_nrns
        
        neuron = self.get_neuron_from_id(neuron_id, neurons_pool)
        max_distance = 0.0
        tip_nrn_id = 0
        for nrn_id in neuron["meta"]["field_list"]:
            crnt_nrn = self.get_neuron_from_id(nrn_id, neurons_pool)
            # calcule la distance entre ce neurone et le centre
            crnt_dist = self.calc_dist(neuron["meta"]["real_center"],crnt_nrn["meta"]["center"])
            if crnt_dist>max_distance:
                max_distance = crnt_dist
                tip_nrn_id = nrn_id
        # calcule les données output
        crnt_nrn = self.get_neuron_from_id(tip_nrn_id, neurons_pool)
        output["local_tip_1"] = crnt_nrn["meta"]["center"]

        # vérifie si la distance avec le tip1 est plus éloigné
        lcl_tip1 = {
            "x":0,
            "y":0
        }
        lcl_tip1["x"] = output["local_tip_1"]["x"] + crnt_nrn["meta"]["vector_1"]["x"]
        lcl_tip1["y"] = output["local_tip_1"]["y"] + crnt_nrn["meta"]["vector_1"]["y"]
        crnt_dist = self.calc_dist(neuron["meta"]["real_center"],lcl_tip1)
        # si c'est le cas, utilise cette nouvelle distance
        if crnt_dist>max_distance:
            max_distance = crnt_dist
            output["local_tip_1"] = lcl_tip1
            output["vector_1"]["y"] = ( crnt_nrn["meta"]["vector_2"]["y"] - crnt_nrn["meta"]["vector_1"]["y"] ) / 2
            output["vector_1"]["x"] = ( crnt_nrn["meta"]["vector_2"]["x"] - crnt_nrn["meta"]["vector_1"]["x"] ) / 2
        else:
            lcl_tip1["x"] = output["local_tip_1"]["x"] + crnt_nrn["meta"]["vector_2"]["x"]
            lcl_tip1["y"] = output["local_tip_1"]["y"] + crnt_nrn["meta"]["vector_2"]["y"]
            crnt_dist = self.calc_dist(neuron["meta"]["real_center"],lcl_tip1)
            # si c'est le cas, utilise cette nouvelle distance
            if crnt_dist>max_distance:
                max_distance = crnt_dist
                output["local_tip_1"] = lcl_tip1
                output["vector_1"]["y"] = -( crnt_nrn["meta"]["vector_2"]["y"] - crnt_nrn["meta"]["vector_1"]["y"] ) / 2
                output["vector_1"]["x"] = -( crnt_nrn["meta"]["vector_2"]["x"] - crnt_nrn["meta"]["vector_1"]["x"] ) / 2
            
        for nrn_id in neuron["meta"]["field_list"]:
            crnt_nrn = self.get_neuron_from_id(nrn_id, neurons_pool)
            crnt_dist = self.calc_dist(output["local_tip_1"],crnt_nrn["meta"]["center"])
            if crnt_dist>max_distance:
                max_distance = crnt_dist
                tip_nrn_id = nrn_id
        # calcule les données output
        crnt_nrn = self.get_neuron_from_id(tip_nrn_id, neurons_pool)
        output["local_tip_2"] = crnt_nrn["meta"]["center"]
        output["vector_2"]["y"] = ( crnt_nrn["meta"]["vector_2"]["y"] - crnt_nrn["meta"]["vector_1"]["y"] ) / 2
        output["vector_2"]["x"] = ( crnt_nrn["meta"]["vector_2"]["x"] - crnt_nrn["meta"]["vector_1"]["x"] ) / 2

        # vérifie si la distance avec le tip1 est plus éloigné
        lcl_tip2 = {
            "x":0,
            "y":0
        }
        lcl_tip2["x"] = output["local_tip_2"]["x"] + crnt_nrn["meta"]["vector_1"]["x"]
        lcl_tip2["y"] = output["local_tip_2"]["y"] + crnt_nrn["meta"]["vector_1"]["y"]
        crnt_dist = self.calc_dist(output["local_tip_1"],lcl_tip2)
        # si c'est le cas, utilise cette nouvelle distance
        if crnt_dist>max_distance:
            max_distance = crnt_dist
            output["local_tip_2"] = lcl_tip2
            output["vector_2"]["y"] = ( crnt_nrn["meta"]["vector_2"]["y"] - crnt_nrn["meta"]["vector_1"]["y"] ) / 2
            output["vector_2"]["x"] = ( crnt_nrn["meta"]["vector_2"]["x"] - crnt_nrn["meta"]["vector_1"]["x"] ) / 2
        else:
            lcl_tip2["x"] = output["local_tip_2"]["x"] + crnt_nrn["meta"]["vector_2"]["x"]
            lcl_tip2["y"] = output["local_tip_2"]["y"] + crnt_nrn["meta"]["vector_2"]["y"]
            crnt_dist = self.calc_dist(output["local_tip_1"],lcl_tip2)
            # si c'est le cas, utilise cette nouvelle distance
            if crnt_dist>=max_distance:
                max_distance = crnt_dist
                output["local_tip_2"] = lcl_tip2
                output["vector_2"]["y"] = -( crnt_nrn["meta"]["vector_2"]["y"] - crnt_nrn["meta"]["vector_1"]["y"] ) / 2
                output["vector_2"]["x"] = -( crnt_nrn["meta"]["vector_2"]["x"] - crnt_nrn["meta"]["vector_1"]["x"] ) / 2
        output["length_c"] = np.round((self.calc_dist(output["local_tip_1"],neuron["meta"]["real_center"])+self.calc_dist(output["local_tip_2"],neuron["meta"]["real_center"])))
        return output

    
    def calc_vector_length(self,vector):
        X_D = pow(vector["x"] ,2)
        Y_D = pow(vector["y"], 2)
        return pow(X_D+Y_D,0.5)


    def get_vector_scalar(self,vector_1, vector_2):
        l1 = self.calc_vector_length(vector_1)
        l2 = self.calc_vector_length(vector_2)
        return l1 * l2 * np.cos(self.calc_angle(vector_1,vector_2))


    def remove_nrn_pos(self, position, neurons_pool=-1):
        lbl_General_Pool = False
        if neurons_pool==-1:
            lbl_General_Pool = True
            neurons_pool = self.lst_nrns
        nrn_id = neurons_pool[position].neuron["_id"]
        layer_id = neurons_pool[position].neuron["layer_id"]

        neurons_pool.pop(position)
        if lbl_General_Pool:
            self.nb_nrns = len(self.lst_nrns)
            self.netGraph.remove_node(nrn_id)
            self.layer_graph[layer_id-1].remove_node(nrn_id)

        for nrn_ctn in neurons_pool:
            nrn = nrn_ctn.neuron
            nrn["DbConnectivity"]["post_synaptique"] = list(set(nrn["DbConnectivity"]["post_synaptique"]))
            nrn["DbConnectivity"]["pre_synaptique"] = list(set(nrn["DbConnectivity"]["pre_synaptique"]))
            for pos in range(len(nrn["DbConnectivity"]["post_synaptique"])):
                if nrn["DbConnectivity"]["post_synaptique"][pos]== nrn_id:
                    nrn["DbConnectivity"]["post_synaptique"].pop(pos)
                    break
            for pos in range(len(nrn["DbConnectivity"]["pre_synaptique"])):
                if nrn["DbConnectivity"]["pre_synaptique"][pos]== nrn_id:
                    nrn["DbConnectivity"]["pre_synaptique"].pop(pos)
                    break
        return self.nb_nrns - 1


    def remove_nrn_by_id(self, nrn_id, neurons_pool=-1):
        lbl_General_Pool = False
        if neurons_pool==-1:
            lbl_General_Pool = True
            neurons_pool = self.lst_nrns
        for nrn_pos in range(len(neurons_pool)):
            if neurons_pool[nrn_pos].neuron["_id"]==nrn_id:
                if lbl_General_Pool:
                    return self.remove_nrn_pos(nrn_pos, -1)
                return self.remove_nrn_pos(nrn_pos, neurons_pool)
        return False
    


    
    def diff_sequence(self, sequence1, sequence2, verbose=False):    
        if sequence1["ratio_pxls_total"]>=sequence2["ratio_pxls_total"]:
            dist_norme = np.arange(0, sequence2["ratio_pxls_total"], 0.01)
            new_seq2 = np.interp(dist_norme, sequence2["ratio_dist"], sequence2["l_angles"])

            dbl_seq1 = sequence1
            dbl_seq2 = sequence2
        else:
            dist_norme = np.arange(0, sequence1["ratio_pxls_total"], 0.01)
            new_seq2 = np.interp(dist_norme, sequence1["ratio_dist"], sequence1["l_angles"])

            dbl_seq1 = sequence2
            dbl_seq2 = sequence1
        # rajouter un offset à dist_norme pour décaler les résultats de séquences la plus longue
        # le max offset est max_offset = sequence1["ratio_pxls_total"] - sequence1["ratio_pxls_total"]
        max_offset = int(np.floor(100*(dbl_seq1["ratio_pxls_total"] - dbl_seq2["ratio_pxls_total"])))
        min_error = -1
        saved_offset = -1
        saved_seq1 = []
        premier = 0
        new_seq1 = np.array([])
        lbl_init = True
        
        for offset in range(0, max_offset +1):
            
            offset = offset/100
            if offset<0:
                continue
            new_seq1 = np.interp(dist_norme+offset, dbl_seq1["ratio_dist"], dbl_seq1["l_angles"])
            # pas de normalisation des angles finalement
            # if np.max(np.abs(new_seq1))!=0:
            #     new_seq1 = new_seq1/np.max(np.abs(new_seq1))
            new_error = np.sum(np.power(new_seq1 - new_seq2,2))
            premier += 1
            if (lbl_init and not np.isnan(new_error)) or (new_error < min_error and not np.isnan(new_error)):
                min_error = new_error
                saved_offset = copy.deepcopy(offset)
                saved_seq1 = new_seq1
                lbl_init = False
        
        if verbose or saved_offset==-1:
            # un premier affichage graphique pour observer les résultats 
            # if premier >= 1:
            _, ax = plt.subplots()
            ax.plot(saved_seq1,"kx-")
            ax.plot(new_seq2,"r*--")
            print("new_seq2",new_seq2)
            str_title = str(premier)+", error: "+str(np.around(min_error,2))+" offset: "+str(saved_offset)
            ax.set_title(str_title)

        return min_error/dbl_seq2["ratio_pxls_total"], saved_offset

    
    def test_sequences(self, sequence1, sequence2, verbose=False):
        result, saved_offset = self.diff_sequence(sequence1, sequence2,verbose)
        # Retourne la séquence
        sequence2["path"] = sequence2["path"][::-1]
        # sequence2["activations"] = sequence2["activations"][::-1]
        A = np.cumsum(sequence2["delta_l"][::-1])[:-1]
        A = sequence2["ratio_pxls_total"]*A/np.max(A)
        sequence2["ratio_dist"] = A
        sequence2["l_angles"] = -1 * sequence2["l_angles"][::-1]
        # sequence2["l_angles"] = sequence2["l_angles"][::-1]
        # Test sur la séquence retournée
        result2, saved_offset2 = self.diff_sequence(sequence1, sequence2,verbose)
        # Remet la séquence comme elle était à l'origine 
        sequence2["path"] = sequence2["path"][::-1]
        # sequence2["activations"] = sequence2["activations"][::-1]
        A = np.cumsum(sequence2["delta_l"])[:-1]
        A = sequence2["ratio_pxls_total"]*A/np.max(A)
        sequence2["ratio_dist"] = A
        sequence2["l_angles"] = -1 * sequence2["l_angles"][::-1]
        # sequence2["l_angles"] = sequence2["l_angles"][::-1]
        
        if result>result2:
            return result2, saved_offset2
        else:
            return result, saved_offset

