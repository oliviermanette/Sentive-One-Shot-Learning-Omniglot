import copy

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import numpy as np
from sklearn.decomposition import PCA

import networkx as nx

from .sentive_neuron_helper import sentive_neuron_helper

class sentive_vision_network(object):

    def __init__(self, episode):

        self.episode = episode
        # print("and here")

        # Affiche le graphique du caractère à reconnaitre
        plt.matshow(episode[:,:,0])
        # Il va s'afficher en dernier avec ma version de jupyter notebook.
        
        ###########################################
        # meta parameters
        self.SEUIL = -0.5

        self.IMG_SIZE = 28
        self.angle_tolerance_deg = 17
        self.ANGL_TOL = np.pi * self.angle_tolerance_deg / 180
        self.angle_tolerance_deg = 17
        self.ANGL_TOL2 = np.pi * self.angle_tolerance_deg / 180

        # si plus petit que EPSILON, considère que c'est égal
        self.ANGL_EPSILON = np.pi * 1 / 180
        
        # POURCENTAGE : MERGE_LIM = 75%
        self.MERGE_LIM = 90
        self.limite_merge = 1 - self.MERGE_LIM/100

        # self.MAX_ANGL = 0.75

        # POURCENTAGE DE PIXELS UNIQUE MINIMUM 
        self.MIN_PIXEL = 10

        self.MIN_PATH = 3

        # end metaparameters
        ###########################################

        # nb est le premier identifiant pour créer les neurones
        self.nb = 0

        # premier identifiant des neurones de la couche 3
        self.nb_min_nrn3 = 0

        # liste contenant tous les neurones : pool_vision
        # self.pool_vision = []
        self.nrn_pxl_map = np.zeros([self.IMG_SIZE,self.IMG_SIZE])
        self.nrn_l2_map = np.zeros([self.IMG_SIZE,self.IMG_SIZE])
        self.np_coord = []

        # fonctions utilitaires
        # neuron_tools 
        self.nrn_tls = sentive_neuron_helper()

        self.glbl_prm = {
            "cg":{"x":0,"y":0},
            "u_axis":{"x":0,"y":0},
            "v_axis":{"x":0,"y":0}
            }
        
        self.nb_ltrl_conn = []

        self.nb_nrn_pxls = 0
        self.nrn_pxls = []

        self.nrn_segments = []
        self.slct_sgmts = []
        self.nrn_saccade = []

        self.curve_prototype = {
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
			"acceleration_step" : 0.0
		}


    def layer_1(self):
        ##################################################
        ######## NEURONES DE LA COUCHE 1 (t_1) #########
        ##################################################
        self.nrn_tls.new_layer()
        # Crée un neurone par pixel au début:
        pxl_coord = []
        for y in range(1,self.IMG_SIZE-1):
            for x in range(1,self.IMG_SIZE-1):
                if self.episode[y][x][0]>self.SEUIL:
                    nb  = self.nrn_tls.add_new_nrn()
                    self.nrn_pxls.append(nb)
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["center"]["x"] = x
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["center"]["y"] = y
                    self.nrn_tls.lst_nrns[nb].neuron["meta"]["matrix_width"] = 1
                    # le poids est la valeur du pixel
                    # passer les valeurs de l'image entre -1 et 1 à des valeurs comprises entre 0 et 1
                    self.nrn_tls.lst_nrns[nb].neuron["weight"] = self.episode[y][x][0] / 2 + 0.5
                    
                    self.nrn_pxl_map[y][x] = self.nrn_tls.lst_nrns[nb].neuron["_id"]

                    pxl_coord.append([x,y])
        
        self.nb_nrn_pxls = copy.deepcopy(self.nrn_tls.nb_nrns)
        print("nombre de neurones taille 1:",self.nb_nrn_pxls)

        pca = PCA(n_components=2)
        pca.fit(pxl_coord)
        # on obtient les résultats ici:
        # print(pca.components_)
        # permet d'avoir l'orientation globale du caractère
        self.glbl_prm["u_axis"]["x"]=pca.components_[0][0]
        self.glbl_prm["u_axis"]["y"]=pca.components_[0][1]
        self.glbl_prm["v_axis"]["x"]=pca.components_[1][0]
        self.glbl_prm["v_axis"]["y"]=pca.components_[1][1]

        # calcule le centre de gravité des pixels
        self.np_coord = np.array(pxl_coord)
        self.glbl_prm["cg"]["x"] = np.mean(self.np_coord[:,0])
        self.glbl_prm["cg"]["y"] = np.mean(self.np_coord[:,1])
        # print(self.glbl_prm)

    
    def layer_2(self):
        ##################################################
        ########## NEURONES DE LA COUCHE 2 (t_3) #########
        ##################################################
        # Les neurones de cette couche ont des champs récepteurs 
        # qui sont des matrices de *3x3* mais orientées

        # copie locale de nrn_pxl_map
        nrn_pxl_map = copy.deepcopy(self.nrn_pxl_map)

        # Création de la nouvelle couche
        self.nrn_tls.new_layer()
        # liste contenant les id de la couche 2
        lst_nrn2_pos = []

        # on modifie les paramètres du neurone créé
        # on crée des positions new_x et new_y vides pour le neurone
        new_x = None
        new_y = None

        # Je crée une variable qui va contenir l'angle de rotation précédent
        previous_angle_a_n2 = None

        # initialisation de composants de décalage du vecteur de déplacement
        shift_x = 0
        shift_y = 0

        # liste de nrn3 
        lst_nrn3 = []

        # liste de nrn3 coupled
        list_coupled_nrn3s = []

        bool_first_nrn2 = True

        lst_nrn3_found = []

        # boucle tant que new_x et new_y sont positifs ou vide:
        while new_x == None or new_y == None or (new_x > 0 and new_y > 0):
            # on crée un nouveau neurone de la couche 2
            nb  = self.nrn_tls.add_new_nrn()
            if bool_first_nrn2:
                # id minimum des neurones de la couche 2
                nb_min_nrn2 = copy.deepcopy(nb)
                bool_first_nrn2 = False
            # on ajoute le neurone à la liste des neurones de la couche 2
            lst_nrn2_pos.append(nb)
            # racourci pour accéder au neurone
            nrn2 = self.nrn_tls.lst_nrns[nb].neuron

            # si new_x et new_y sont vides on prend la position du premier neurone de la couche 1
            if new_x == None or new_y == None:
                new_x = self.nrn_tls.lst_nrns[0].neuron["meta"]["center"]["x"]
                nrn2["meta"]["center"]["x"] = new_x
                new_y = self.nrn_tls.lst_nrns[0].neuron["meta"]["center"]["y"]
                nrn2["meta"]["center"]["y"] = new_y
                nrn2["meta"]["matrix_width"] = 3
            else:
                nrn2["meta"]["center"]["x"] = new_x
                nrn2["meta"]["center"]["y"] = new_y
                nrn2["meta"]["matrix_width"] = 3
            print("\n\n______________________________________________________________________________________")
            print("nrn2 id", nrn2["_id"])
            print("new_x", new_x)
            print("new_y", new_y)
            # pixel central du neurone
            central_pixel_id = int(self.nrn_pxl_map[new_y][new_x])
            print("central_pixel_id", central_pixel_id)

            # supprime le pixel central de la map
            print("nrn_pxl_map[new_y][x]", nrn_pxl_map[new_y][new_x])
            nrn_pxl_map[new_y][new_x] = 0

            # tmp_sub_pxl_map contient les identifiants de chaque neurone pixel sur une carte nrnl_map
            tmp_sub_pxl_map = nrn_pxl_map[new_y - 1 : new_y + 2, new_x-1:new_x+2]
            print("tmp_sub_pxl_map :\n", tmp_sub_pxl_map)

            # si tmp_sub_pxl_map ne contient que des 0 on quitte la boucle
            if np.count_nonzero(tmp_sub_pxl_map) == 0:
                print("\n continue seulement des zéros\n")
                new_x = None
                new_y = None
                # vérifie si il reste des neurones pixels dans la map nrn_pxl_map
                if np.count_nonzero(nrn_pxl_map) == 0:
                    print("\n break plus de neurones pixels\n")
                    break
                else:
                    print("\n continue il reste des neurones pixels\n")
                    # trouve le premier neurone pixel dans la map
                    for tmp_y in range(0, nrn_pxl_map.shape[0]):
                        for tmp_x in range(0, nrn_pxl_map.shape[1]):
                            if nrn_pxl_map[tmp_y][tmp_x] > 0:
                                new_x = tmp_x
                                new_y = tmp_y
                                print("new_x", new_x)
                                print("new_y", new_y)
                                # pixel central du neurone
                                central_pixel_id = int(self.nrn_pxl_map[new_y][new_x])
                                nrn2["meta"]["center"]["x"] = new_x
                                nrn2["meta"]["center"]["y"] = new_y
                                nrn_pxl_map[new_y][new_x] = 0
                                # tmp_sub_pxl_map contient les identifiants de chaque neurone pixel sur une carte nrnl_map
                                tmp_sub_pxl_map = nrn_pxl_map[new_y - 1 : new_y + 2, new_x-1:new_x+2]
                                print("tmp_sub_pxl_map :\n", tmp_sub_pxl_map)
                                # initialisation de composants de décalage du vecteur de déplacement
                                shift_x = 0
                                shift_y = 0
                                break
                        if new_x != None and new_y != None:
                            break

            nrn2["meta"]["sub_pxl_map"] = self.nrn_pxl_map[new_y-1:new_y+2, new_x-1:new_x+2]

            tmp_list_sub_pxl = list(set(tmp_sub_pxl_map.ravel()))
            # convertir tmp_list_sub_pxl en entiers
            tmp_list_sub_pxl = [int(i) for i in tmp_list_sub_pxl]

            # supprime les neurones pixels dans nrn_pxl_map qui sont dans le champ récepteur du neurone
            for tmp_y in range(new_y-1,new_y+2):
                for tmp_x in range(new_x-1,new_x+2):
                    nrn_pxl_map[tmp_y][tmp_x] = 0

            #     ___      _         _        _   _          
            #    / _ \ _ _(_)___ _ _| |_ __ _| |_(_)___ _ _  
            #   | (_) | '_| / -_) ' \  _/ _` |  _| / _ \ ' \ 
            #    \___/|_| |_\___|_||_\__\__,_|\__|_\___/_||_|
            #                                                
            ## calcule le vecteur d'orientation moyen des pixels
            # pour chaque neurone pixel de la liste tmp_sub_pxl_map on calcule le vecteur d'orientation
            x_composant = []
            y_composant = []
            lst_nrn_pxl_pos = []
            mtrx_weights_pxl = np.zeros([3,3])
            weight_sum = 0
            # poids du pixel central
            mtrx_weights_pxl[1][1] = self.nrn_tls.lst_nrns[central_pixel_id-1].neuron["weight"] 

            print("tmp_list_sub_pxl", tmp_list_sub_pxl)
            for nrn_pxl_id in tmp_list_sub_pxl:
                if nrn_pxl_id > 0:
                    print("nrn_pxl_id", nrn_pxl_id)
                    # on récupère le neurone pixel
                    nrn_pxl = self.nrn_tls.lst_nrns[nrn_pxl_id-1].neuron
                    # on pondère par le poids du neurone pixel
                    weight_sum += nrn_pxl["weight"]
                    print("nrn_pxl id & weight :", nrn_pxl["_id"],nrn_pxl["weight"])
                    # on récupère le vecteur d'orientation du neurone pixel
                    x_composant.append(nrn_pxl["weight"]*(nrn_pxl["meta"]["center"]["x"] - nrn2["meta"]["center"]["x"]))
                    y_composant.append(nrn_pxl["weight"]*(nrn_pxl["meta"]["center"]["y"] - nrn2["meta"]["center"]["y"]))
                    # on met dans une liste les coordonnées des neurones pixels
                    lst_nrn_pxl_pos.append(nrn_pxl["meta"]["center"])
                    mtrx_weights_pxl[nrn_pxl["meta"]["center"]["y"] - nrn2["meta"]["center"]["y"] + 1][nrn_pxl["meta"]["center"]["x"] - nrn2["meta"]["center"]["x"] + 1] = nrn_pxl["weight"]

            # ajoute les id des neurones pixels dans la liste des pre_synaptique
            nrn2["DbConnectivity"]["pre_synaptique"] = tmp_list_sub_pxl
            # ajoute le neurone central dans la liste des pre_synaptique
            nrn2["DbConnectivity"]["pre_synaptique"].append(central_pixel_id)

            # on fait la moyenne des composantes
            print("x_composant", x_composant)
            x_composant = np.sum(x_composant)/weight_sum
            print("y_composant", y_composant)
            y_composant = np.sum(y_composant)/weight_sum
            print("vecteur orientation sans shift:",x_composant,y_composant)

            # on calcule le décalage du vecteur d'orientation
            # x_composant = x_composant - shift_x
            # y_composant = y_composant - shift_y
            # print("shift_x", shift_x)
            # print("shift_y", shift_y)
            # print("vecteur orientation AVEC shift:",x_composant,y_composant)
            # print("POSITION REELLE FINALE", new_x + x_composant, new_y + y_composant)
            desired_x = np.floor(new_x + x_composant)
            desired_y = np.floor(new_y + y_composant)

            # on sauvegarde le vecteur d'orientation
            nrn2["meta"]["orientation"] = {
                "x": x_composant,
                "y": y_composant
            }

            offset_x = np.round(x_composant)
            offset_y = np.round(y_composant)

            # calcul du décalage du vecteur d'orientation
            shift_x += x_composant - offset_x
            shift_y += y_composant - offset_y

            # calcul de la position du prochain neurone de la couche 2
            new_new_x = int(np.ceil(nrn2["meta"]["center"]["x"] + offset_x))
            new_new_y = int(np.ceil(nrn2["meta"]["center"]["y"] + offset_y))

            if new_new_x != new_x or new_new_y != new_y:
                new_x = new_new_x
                new_y = new_new_y
                print("PROCHAIN NEURONE: [new_x:", new_x, ", new_y:", new_y, "]")
            else:
                if x_composant < 0:
                    print("x_composant < 0")
                    print("x_composant", x_composant)
                    print("np.floor(x_composant)", np.floor(x_composant))
                    new_x = int(np.round(nrn2["meta"]["center"]["x"] + np.floor(x_composant)))
                else:
                    print("x_composant > 0")
                    print("x_composant", x_composant)
                    print("np.ceil(x_composant)", np.ceil(x_composant))
                    new_x = int(np.round(nrn2["meta"]["center"]["x"] + np.ceil(x_composant)))
                if y_composant < 0:
                    new_y = int(np.round(nrn2["meta"]["center"]["y"] + np.floor(y_composant)))
                else:
                    new_y = int(np.round(nrn2["meta"]["center"]["y"] + np.ceil(y_composant)))

            print("PROCHAIN NEURONE SANS CHANGEMENT: [new_x:", new_x, ", new_y:", new_y, "]")
            print("mtrx_weights_pxl:\n", mtrx_weights_pxl)
            # prochain neurone new_x et new_y dans les coordonnées de mtrx_weights_pxl
            print("neurone a supprimer dans la matrice : x :", new_x - nrn2["meta"]["center"]["x"] + 1, "y:", new_y - nrn2["meta"]["center"]["y"] + 1)

            # on supprime le neurone suivant de la matrice afin de calculer un poids des neurones de base
            mtrx_weights_pxl[new_y - nrn2["meta"]["center"]["y"] + 1][new_x - nrn2["meta"]["center"]["x"] + 1] = 0
            print("mtrx_weights_pxl:\n", mtrx_weights_pxl)

            # calcul des coordonnées du barycentre de la mtrx_weights_pxl
            # on calcule le barycentre de la matrice
            barycentre = np.zeros([2])
            for y in range(0,3):
                for x in range(0,3):
                    barycentre[0] += x * mtrx_weights_pxl[y][x]
                    barycentre[1] += y * mtrx_weights_pxl[y][x]

            barycentre[0] = barycentre[0] / np.sum(mtrx_weights_pxl)
            barycentre[1] = barycentre[1] / np.sum(mtrx_weights_pxl)
            print("barycentre", barycentre)

            # on vérifie que le neurone est dans la carte
            if new_x < 0 or new_x >= self.nrn_pxl_map.shape[1] or new_y < 0 or new_y >= self.nrn_pxl_map.shape[0]:
                print("\nNeurone hors de la carte")
                new_x = -1
                new_y = -1

            #                      _____  
            #                     |____ | 
            #   _ __  _ __ _ __       / / 
            #  | '_ \| '__| '_ \      \ \ 
            #  | | | | |  | | | | .___/ / 
            #  |_| |_|_|  |_| |_| \____/  
            #                             
            #                                                 
            # variable bool qui sert à savoir si on a trouvé un neurone valide de la couche 3              
            nrn3_not_found = True
            nrn3_curve_not_found = True

            # diretions détectées 
            new_pixels_detected = []

            # liste temporaire des neurones nrn3 trouvés
            tmp_lst_nrn3_found = []
            
            # boucle sur chaque nrn3 dans la liste :
            for nb3 in lst_nrn3:
                # racourci pour accéder au neurone
                nrn3 = self.nrn_tls.lst_nrns[nb3].neuron

                #                    ____  _ _          
                #    _ _  _ _ _ _   |__ / | (_)_ _  ___ 
                #   | ' \| '_| ' \   |_ \ | | | ' \/ -_)
                #   |_||_|_| |_||_| |___/ |_|_|_||_\___|
                #                                       
                # si le nrn3 est une ligne :
                if nrn3["type"] == "sentive_vision_line":
                    print("******************")
                    print("nrn3", nrn3["_id"], "est une ligne")
                    # on récupère le basis_vector du neurone 3
                    basis_vector = nrn3["meta"]["line"]["basis_vector"]
                    # on calcul la norme du basis vector
                    norm_basis_vector = np.sqrt(np.power(basis_vector["x"],2)+np.power(basis_vector["y"],2))
                    # on normalise le basis_vector
                    if norm_basis_vector != 0:
                        basis_vector = {
                                "x": basis_vector["x"] / norm_basis_vector,
                                "y": basis_vector["y"] / norm_basis_vector
                        }

                    # on récupère le starting_point du neurone 3
                    starting_point = nrn3["meta"]["line"]["starting_point"]

                    # on fait une boucle jusqu'à ce qu'on trouve la position actuelle. Si la distance augmente on sort de la boucle.
                    # on initialise la distance entre le starting_point et la position actuelle
                    distance = np.sqrt(np.power(starting_point["x"] - nrn2["meta"]["center"]["x"],2)+np.power(starting_point["y"] - nrn2["meta"]["center"]["y"],2))
                    print("distance initiale:", distance)
                    min_distance = copy.deepcopy(distance)

                    # on initialise le new_point
                    new_point = copy.deepcopy(starting_point)
                    # compteur de boucle
                    i = 0
                    # boucle qui s'arrête si la distance augmente ou si elle devient inférieure à 1
                    while distance == min_distance :
                        # détermination du new_point
                        new_point = {
                            "x": new_point["x"] + basis_vector["x"],
                            "y": new_point["y"] + basis_vector["y"]
                        }
                        # calcul de la distance entre le new_point et le nrn2["meta"]["center"]
                        distance = np.sqrt(np.power(new_point["x"] - nrn2["meta"]["center"]["x"],2)+np.power(new_point["y"] - nrn2["meta"]["center"]["y"],2))
                        # on incrémente le compteur de boucle
                        i += 1
                        # on M.A.J. la distance minimale
                        if distance < min_distance:
                            min_distance = copy.deepcopy(distance)
                        if distance < 1:
                            break
                        if i > 100:
                            break

                    # on affiche le nombre de boucle et le new_point
                    print("nombre de boucle:", i, ", new_point:", new_point, ", center", nrn2["meta"]["center"])
                    print("* distance finale:", distance, ', (min_distance:', min_distance, ')')
                    
                    pos_predict = []
                    # si la distance est nulle, on peut faire une prédiction
                    if distance < 1:
                        nrn3_pos_predict = {
                            "x": np.floor(new_point["x"] + basis_vector["x"]),
                            "y": np.floor(new_point["y"] + basis_vector["y"])
                        }
                        pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                        nrn3_pos_predict = {
                            "x": np.round(new_point["x"] + basis_vector["x"]),
                            "y": np.round(new_point["y"] + basis_vector["y"])
                        }
                        pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                        nrn3_pos_predict = {
                            "x": np.ceil(new_point["x"] + basis_vector["x"]),
                            "y": np.ceil(new_point["y"] + basis_vector["y"])
                        }
                        pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                        nrn3_pos_predict = {
                            "x": np.floor(new_point["x"] + 1.5 * basis_vector["x"]),
                            "y": np.floor(new_point["y"] + 1.5 * basis_vector["y"])
                        }
                        pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                        nrn3_pos_predict = {
                            "x": np.round(new_point["x"] + 1.5 * basis_vector["x"]),
                            "y": np.round(new_point["y"] + 1.5 * basis_vector["y"])
                        }
                        pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                        nrn3_pos_predict = {
                            "x": np.ceil(new_point["x"] + 1.5 * basis_vector["x"]),
                            "y": np.ceil(new_point["y"] + 1.5 * basis_vector["y"])
                        }
                        pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                    else:
                        nrn3_pos_predict = {
                            "x": np.floor(new_point["x"]),
                            "y": np.floor(new_point["y"])
                        }
                        pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                        nrn3_pos_predict = {
                            "x": np.round(new_point["x"]),
                            "y": np.round(new_point["y"])
                        }
                        pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                        new_point = {
                            "x": new_point["x"] - basis_vector["x"],
                            "y": new_point["y"] - basis_vector["y"]
                        }
                    
                    # alternative 1
                    nrn3_pos_predict = {
                        "x": np.floor(pos_predict[0]["x"] - basis_vector["y"]),
                        "y": np.floor(pos_predict[0]["y"] + basis_vector["x"])
                    }
                    pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                    nrn3_pos_predict = {
                        "x": np.round(pos_predict[0]["x"] - basis_vector["y"]),
                        "y": np.round(pos_predict[0]["y"] + basis_vector["x"])
                    }
                    pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                    # alternative 2
                    nrn3_pos_predict = {
                        "x": np.floor(pos_predict[0]["x"] + basis_vector["y"]),
                        "y": np.floor(pos_predict[0]["y"] - basis_vector["x"])
                    }
                    pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                    nrn3_pos_predict = {
                        "x": np.round(pos_predict[0]["x"] + basis_vector["y"]),
                        "y": np.round(pos_predict[0]["y"] - basis_vector["x"])
                    }
                    pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                    
                    print("LINE pos_predict:", pos_predict)

                    ###############################################
                    # Comparaison des pixels alentours avec les pixels prédits
                    ###############################################
                    tmp_new_pixels_detected = {}
                    tmp_angle_min = -999
                    is_found = False
                    orientation = None

                    for coord_pixel in lst_nrn_pxl_pos:
                        test_x = coord_pixel["x"]
                        test_y = coord_pixel["y"]
                        for nrn3_pos_predicted in pos_predict:
                            if abs(nrn3_pos_predicted["x"] - test_x) < 1 and abs(nrn3_pos_predicted["y"] - test_y) < 1:
                                is_found = True
                                
                                tmp_orientation = {
                                    "x": test_x - new_point["x"],
                                    "y": test_y - new_point["y"]
                                }
                                '''
                                orientation = {
                                    "x": test_x - nrn2["meta"]["center"]["x"],
                                    "y": test_y - nrn2["meta"]["center"]["y"]
                                }
                                '''
                                # calcul l'angle entre l'orientation et le vecteur de base
                                tmp_angle = np.abs(self.nrn_tls.calc_angle(tmp_orientation, basis_vector))

                                if tmp_angle_min == -999 or tmp_angle < tmp_angle_min:
                                    orientation = copy.deepcopy(tmp_orientation)
                                    tmp_angle_min = tmp_angle
                                    tmp_new_pixels_detected = {
                                        "x": test_x,
                                        "y": test_y,
                                        "nb_iteration": nrn3["meta"]["line"]["nb_iteration"],
                                        "neuron_type": nrn3["type"],
                                        "distance": copy.deepcopy(min_distance),
                                        "angle": copy.deepcopy(tmp_angle)
                                    }
                                    print("__>")
                                    print("tmp_new_pixels_detected", tmp_new_pixels_detected)
                                    print("nrn3 pos found", nrn3_pos_predicted)
                                    print("angle", tmp_angle)
                                    print("<__")
                                
                    if is_found:
                        new_pixels_detected.append(copy.deepcopy(tmp_new_pixels_detected))

                    if is_found and orientation is not None and min_distance < 1.5:
                        # tmp_lst_nrn3_found.append(nrn3["_id"])
                        # normalisation de l'orientation
                        orientation["x"] = orientation["x"] / np.sqrt(np.power(orientation["x"],2)+np.power(orientation["y"],2))
                        orientation["y"] = orientation["y"] / np.sqrt(np.power(orientation["x"],2)+np.power(orientation["y"],2))

                        # nrn3["meta"]["line"]["nb_iteration"] += 1
                        nrn3["meta"]["pending_nb_iteration"] += 1
                        print("nrn3 ligne trouvé !!!!!!!!!!!!!")
                        # je vais faire une moyenne itérative avec la basis_vector du neurone 3
                        # je récupère le basis_vector du neurone 3
                        print("basis_vector AVANT", nrn3["meta"]["line"]["basis_vector"])
                        tmp_bsvct = copy.deepcopy(nrn3["meta"]["line"]["basis_vector"])
                        # nombre d'itération
                        n = nrn3["meta"]["line"]["nb_iteration"] + 1
                        # je calcule la moyenne itérative

                        nrn3_basis_vector_new_tmp = {
                            "x": self.nrn_tls.FctIterMean(n, orientation["x"], nrn3["meta"]["line"]["basis_vector"]["x"] ),
                            "y": self.nrn_tls.FctIterMean(n, orientation["y"], nrn3["meta"]["line"]["basis_vector"]["y"] )
                        }
                        # nrn3["meta"]["line"]["basis_vector"]["x"] = self.nrn_tls.FctIterMean(n, orientation["x"], nrn3["meta"]["line"]["basis_vector"]["x"] ) 
                        # nrn3["meta"]["line"]["basis_vector"]["y"] = self.nrn_tls.FctIterMean(n, orientation["y"], nrn3["meta"]["line"]["basis_vector"]["y"] )

                        print("--> nouvelle orientation", orientation)
                        # calc_angle de l'ancien et du nouveau basis_vector
                        tmp_angle_bsvct = self.nrn_tls.calc_angle(tmp_bsvct, nrn3_basis_vector_new_tmp)
                        print("basis_vector APRES", nrn3["meta"]["line"]["basis_vector"], "(angle avant et après:", tmp_angle_bsvct, ")")
                        
                        # calculer l'angle max possible             
                        tmp_angle_max = 2 * np.arccos(n/np.sqrt(np.power(n,2)+np.power(1,2)))
                        print("tmp_angle_max ", tmp_angle_max)
                        # si tmp_angle_max >= tmp_angle_bsvct + nrn3["meta"]["cumulated_angle"] alors on valide la ligne
                        if tmp_angle_max >= tmp_angle_bsvct + nrn3["meta"]["cumulated_angle"]:
                            nrn3["meta"]["line"]["nb_iteration"] += 1
                            nrn3["meta"]["cumulated_angle"] += tmp_angle_bsvct
                            # ajout de l'id du nrn2 dans presynaptic DbConnectivity': {'pre_synaptique
                            nrn3["DbConnectivity"]["pre_synaptique"].append(nrn2["_id"])
                            # ajout de l'id du nrn3 dans postsynaptic DbConnectivity': {'post_synaptique
                            nrn2["DbConnectivity"]["post_synaptique"].append(nrn3["_id"])
                            # sauvegarde du basis_vector
                            nrn3["meta"]["line"]["basis_vector"] = nrn3_basis_vector_new_tmp
                            print("angle OK pour sauvegarder ")
                            # finalement on sauvegarde pas
                            nrn3_not_found = False
                    else:
                        nrn3["meta"]["nb_points_aligned"] = 0
                        nrn3["meta"]["pending_nb_iteration"] += 1

                #                  ____                  _ 
                #    _ _  _ _ _ _ |__ /  __ ___ _  _ _ _| |__  ___ ___
                #   | ' \| '_| ' \ |_ \ / _/ _ \ || | '_| '_ \/ -_|_-<
                #   |_||_|_| |_||_|___/ \__\___/\_,_|_| |_.__/\___/__/
                #                                       
                # 
                elif nrn3["type"] == "sentive_vision_curve":
                    print("\n******************")
                    print("nrn3", nrn3["_id"], "est une courbe")
                    print(nrn3["meta"]["curve"]["basis_vector"], "basis_vector")
                    print("nrn3", nrn3["_id"], "nb_iteration", nrn3["meta"]["curve"]["nb_iteration"])
                    # angle
                    print("nrn3", nrn3["_id"], "angle", nrn3["meta"]["curve"]["angle"])
                            
                    if nrn3["meta"]["curve"]["nb_iteration"] == 1:
                        # je récupère le vecteur d'orientation du neurone 2
                        orientation = nrn2["meta"]["orientation"]
                        # je récupère le basis_vector du neurone 3
                        basis_vector = nrn3["meta"]["curve"]["basis_vector"]
                        # je calcule l'angle entre le basis_vector et l'orientation
                        angle = self.nrn_tls.calc_angle(basis_vector, orientation)
                        print("found angle", angle)
                        # je sauvegarde la valeur d'angle dans nrn3
                        nrn3["meta"]["curve"]["angle"] = angle
                        #  je met le compteur nb_d'itération à 2.
                        nrn3["meta"]["curve"]["nb_iteration"] = 2
                        nrn3["meta"]["pending_nb_iteration"] += 1
                        # Je sauvegarde le vecteur d'orientation de nrn2 dans last_vector.
                        nrn3["meta"]["curve"]["last_vector"] = nrn2["meta"]["orientation"]
                        nrn3_not_found = False
                        nrn3_curve_not_found = False
                        # ajout de l'id du nrn2 dans presynaptic DbConnectivity': {'pre_synaptique
                        nrn3["DbConnectivity"]["pre_synaptique"].append(nrn2["_id"])
                        # ajout de l'id du nrn3 dans postsynaptic DbConnectivity': {'post_synaptique
                        nrn2["DbConnectivity"]["post_synaptique"].append(nrn3["_id"])

                    # Si le nombre d'itération est > 1  alors je calcule l'angle entre le basis_vector et l'angle d'orientation du nrn2,
                    elif nrn3["meta"]["curve"]["nb_iteration"] >= 2 and nrn3["meta"]["pending_nb_iteration"] == nrn3["meta"]["curve"]["nb_iteration"]:
                        nrn3["meta"]["pending_nb_iteration"] += 1

                        # on récupère le starting_point du neurone 3
                        starting_point = nrn3["meta"]["curve"]["starting_point"]

                        # on récupère le basis_vector du neurone 3
                        basis_vector = copy.deepcopy(nrn3["meta"]["curve"]["basis_vector"])

                        # on fait une boucle jusqu'à ce qu'on trouve la position actuelle. Si la distance augmente on sort de la boucle.
                        # on initialise la distance entre le starting_point et la position actuelle
                        distance = np.sqrt(np.power(starting_point["x"] - nrn2["meta"]["center"]["x"],2)+np.power(starting_point["y"] - nrn2["meta"]["center"]["y"],2))
                        print("distance initiale:", distance)
                        min_distance = copy.deepcopy(distance)

                        # on initialise le new_point
                        new_point = copy.deepcopy(starting_point)
                        # compteur de boucle
                        i = 0
                        # boucle qui s'arrête si la distance augmente ou si elle devient inférieure à 1
                        while distance == min_distance :
                            if i > 0:
                                # rotation du basis vector (pas tout de suite)
                                basis_vector = self.nrn_tls.draw_rotate_vector(basis_vector, nrn3["meta"]["curve"]["angle"])
                                print("new direction basis vector", basis_vector)

                            # détermination du new_point
                            new_point = {
                                "x": new_point["x"] + basis_vector["x"],
                                "y": new_point["y"] + basis_vector["y"]
                            }
                            print("new point", new_point)

                            # calcul de la distance entre le new_point et le nrn2["meta"]["center"]
                            distance = np.sqrt(np.power(new_point["x"] - nrn2["meta"]["center"]["x"],2)+np.power(new_point["y"] - nrn2["meta"]["center"]["y"],2))
                            print("distance:", distance)
                            # on incrémente le compteur de boucle
                            i += 1
                            # on M.A.J. la distance minimale
                            if distance < min_distance:
                                min_distance = copy.deepcopy(distance)
                            if distance < 1:
                                break
                            if i > 100:
                                break


                        ###############################################
                        # on affiche le nombre de boucle et le new_point
                        print("CURVED nombre de boucles:", i, ", new_point:", new_point, ", center", nrn2["meta"]["center"])
                        print("* distance finale:", distance, ', (min_distance:', min_distance, ')')
                        

                        #  |)   _  |. _|-.  ,_
                        #  | |`(/_(||(_|_|()||
                        # 
                        
                        check_predict = True
                        while check_predict:
                            pos_predict = []
                            # si la distance est nulle, on peut faire une prédiction
                            if distance < 1:
                                vector_predicted = self.nrn_tls.draw_rotate_vector(basis_vector, nrn3["meta"]["curve"]["angle"])
                                nrn3_pos_predict = {
                                    "x": np.floor(new_point["x"] + vector_predicted["x"]),
                                    "y": np.floor(new_point["y"] + vector_predicted["y"])
                                }
                                # calcule entre la position prédite et le nrn2["meta"]["center"]
                                distance = np.sqrt(np.power(nrn3_pos_predict["x"] - nrn2["meta"]["center"]["x"],2)+np.power(nrn3_pos_predict["y"] - nrn2["meta"]["center"]["y"],2))
                                nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                                pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                                nrn3_pos_predict = {
                                    "x": np.round(new_point["x"] + vector_predicted["x"]),
                                    "y": np.round(new_point["y"] + vector_predicted["y"])
                                }
                                nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                                pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                                nrn3_pos_predict = {
                                    "x": np.ceil(new_point["x"] + vector_predicted["x"]),
                                    "y": np.ceil(new_point["y"] + vector_predicted["y"])
                                }
                                nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                                pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                                nrn3_pos_predict = {
                                    "x": np.floor(new_point["x"] + 1.5 * vector_predicted["x"]),
                                    "y": np.floor(new_point["y"] + 1.5 * vector_predicted["y"])
                                }
                                nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                                pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                                nrn3_pos_predict = {
                                    "x": np.round(new_point["x"] + 1.5 * vector_predicted["x"]),
                                    "y": np.round(new_point["y"] + 1.5 * vector_predicted["y"])
                                }
                                nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                                pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                                nrn3_pos_predict = {
                                    "x": np.ceil(new_point["x"] + 1.5 * vector_predicted["x"]),
                                    "y": np.ceil(new_point["y"] + 1.5 * vector_predicted["y"])
                                }
                                nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                                pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                            else: # sinon on fait une prédiction sur la position du new_point (dépassement de la position de distance minimale)
                                nrn3_pos_predict = {
                                    "x": np.floor(new_point["x"]),
                                    "y": np.floor(new_point["y"])
                                }
                                nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                                pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                                nrn3_pos_predict = {
                                    "x": np.round(new_point["x"]),
                                    "y": np.round(new_point["y"])
                                }
                                nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                                pos_predict.append(copy.deepcopy(nrn3_pos_predict))

                                # Je recule d'un élément car dans ce cas, cela signifie qu'on a dépassé la position de distance minimale
                                new_point = {
                                    "x": new_point["x"] - basis_vector["x"],
                                    "y": new_point["y"] - basis_vector["y"]
                                }
                                vector_predicted = copy.deepcopy(basis_vector)
                                basis_vector = self.nrn_tls.draw_rotate_vector(basis_vector, -nrn3["meta"]["curve"]["angle"])
                            
                            # alternative 1
                            nrn3_pos_predict = {
                                "x": np.floor(pos_predict[0]["x"] - vector_predicted["y"]),
                                "y": np.floor(pos_predict[0]["y"] + vector_predicted["x"])
                            }
                            nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                            pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                            nrn3_pos_predict = {
                                "x": np.round(pos_predict[0]["x"] - vector_predicted["y"]),
                                "y": np.round(pos_predict[0]["y"] + vector_predicted["x"])
                            }
                            nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                            pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                            # alternative 2
                            nrn3_pos_predict = {
                                "x": np.floor(pos_predict[0]["x"] + vector_predicted["y"]),
                                "y": np.floor(pos_predict[0]["y"] - vector_predicted["x"])
                            }
                            nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                            pos_predict.append(copy.deepcopy(nrn3_pos_predict))
                            nrn3_pos_predict = {
                                "x": np.round(pos_predict[0]["x"] + vector_predicted["y"]),
                                "y": np.round(pos_predict[0]["y"] - vector_predicted["x"])
                            }
                            nrn3_pos_predict = self.test_pos_predict_on_3x3_grid(nrn2, nrn3_pos_predict)
                            pos_predict.append(copy.deepcopy(nrn3_pos_predict))

                            print("CURVE vector_predicted", vector_predicted)
                            print("CURVE pos_predict:", pos_predict)
                            print("========================")
                            print("CURVE nrn3 id",nrn3["_id"] , "CURVE position", nrn3["meta"]["curve"]["nb_iteration"], "->", nrn3["meta"]["curve"]["nb_iteration"] + 1)
                            print("========================")

                            
                            ###############################################################
                            # comparer les positions prédites avec chacune des positions des neurones pixels dans la liste lst_nrn_pxl_pos
                            tmp_new_pixels_detected = {}
                            tmp_angle_min = -999
                            tmp_angle_min_alt2 = -999
                            test_angle_min = -999
                            is_found = False
                            orientation = None
                            for coord_pixel in lst_nrn_pxl_pos:
                                test_x = coord_pixel["x"]
                                test_y = coord_pixel["y"]
                                for nrn3_pos_predicted in pos_predict:
                                    if abs(nrn3_pos_predicted["x"] - test_x) < 1 and abs(nrn3_pos_predicted["y"] - test_y) < 1:
                                        is_found = True

                                        tmp_orientation = {
                                            "x": test_x - new_point["x"],
                                            "y": test_y - new_point["y"]
                                        }
                                        tmp_orientation_alt2 = {
                                            "x": test_x - nrn2["meta"]["center"]["x"],
                                            "y": test_y - nrn2["meta"]["center"]["y"]
                                        }
                                        tmp_angle = self.nrn_tls.calc_angle(basis_vector, tmp_orientation)
                                        test_angle = np.abs(self.nrn_tls.calc_angle(vector_predicted, tmp_orientation))
                                        tmp_angle_alt2 = self.nrn_tls.calc_angle(basis_vector, tmp_orientation_alt2)
                                        print("CURVE tmp_orientation", tmp_orientation)
                                        print("CURVE vector_predicted", vector_predicted)
                                        print("CURVE tmp_angle", tmp_angle)
                                        print("CURVE test_angle", test_angle)
                                        if test_angle_min == -999 or test_angle < test_angle_min:
                                            test_angle_min = test_angle
                                            orientation = copy.deepcopy(tmp_orientation)
                                            tmp_angle_min = tmp_angle
                                            tmp_angle_min_alt2 = tmp_angle_alt2
                                            tmp_new_pixels_detected = {
                                                "x": test_x,
                                                "y": test_y,
                                                "nb_iteration": nrn3["meta"]["curve"]["nb_iteration"],
                                                "neuron_type": nrn3["type"],
                                                "distance" : copy.deepcopy(min_distance),
                                                "angle" : self.nrn_tls.calc_angle(tmp_orientation, vector_predicted)
                                            }
                                            print("__>")
                                            print("CURVED tmp_new_pixels_detected", tmp_new_pixels_detected)
                                            print("nrn3 pos found", nrn3_pos_predicted)
                                            print("angle", tmp_angle)
                                            print("<__")
                            if is_found:
                                new_pixels_detected.append(copy.deepcopy(tmp_new_pixels_detected))
                                check_predict = False
                            else:
                                if distance < 1:
                                    new_point = {
                                        "x": new_point["x"] + vector_predicted["x"],
                                        "y": new_point["y"] + vector_predicted["y"]
                                    }
                                    # calcule entre la position prédite et le nrn2["meta"]["center"]
                                    distance = np.sqrt(np.power(nrn3_pos_predict["x"] - nrn2["meta"]["center"]["x"],2)+np.power(nrn3_pos_predict["y"] - nrn2["meta"]["center"]["y"],2))
                                    print("check_predict distance:", distance, "min_distance:", min_distance, "new_point:", new_point)
                                    if distance < min_distance:
                                        min_distance = copy.deepcopy(distance)
                                        check_predict = True
                                    else:
                                        check_predict = False
                                else:
                                    check_predict = False

                        if is_found and orientation is not None and min_distance < 1.5:
                            tmp_lst_nrn3_found.append(nrn3["_id"])
                            nrn3_not_found = False
                            nrn3_curve_not_found = False
                            print("Prédiction nrn3 CURVE", nrn3["_id"], "TROUVé")
                            print("************************************")
                            nrn3["meta"]["curve"]["nb_iteration"] += 1
                            # mettre à jour l'angle
                            # angle = self.nrn_tls.calc_angle(nrn3["meta"]["curve"]["last_vector"], orientation)
                            
                            print("vecteur d'orientation Choisi", orientation)
                            print("nouvel angle mesuré", tmp_angle_min, "(alt2 centre:", tmp_angle_min_alt2, ")")
                            angle = self.nrn_tls.FctIterMean(nrn3["meta"]["curve"]["nb_iteration"], tmp_angle_min, nrn3["meta"]["curve"]["angle"])
                            print("nb itérations:", nrn3["meta"]["curve"]["nb_iteration"])
                            print("=== angle M.à.J.", angle, "===")
                            # je sauvegarde la valeur d'angle dans nrn3
                            nrn3["meta"]["curve"]["angle"] = angle
                            # Je sauvegarde le vecteur d'orientation de nrn2 dans last_vector.
                            nrn3["meta"]["curve"]["last_vector"] = orientation
                            # ajout de l'id du nrn2 dans presynaptic DbConnectivity': {'pre_synaptique
                            nrn3["DbConnectivity"]["pre_synaptique"].append(nrn2["_id"])
                            # ajout de l'id du nrn3 dans postsynaptic DbConnectivity': {'post_synaptique
                            nrn2["DbConnectivity"]["post_synaptique"].append(nrn3["_id"])

            # si on a pas trouvé de neurone de la couche 3 valide on en crée un nouveau
            # un nrn ligne
            if nrn3_not_found:
                # je crée un couple de neurone de la couche 3, un neurone ligne et un neurone courbe
                # le neurone courbe :
                lst_nrn3 = self.create_new_nrn3(lst_nrn3, nrn2)
                new_nrn3_curve = self.nrn_tls.lst_nrns[lst_nrn3[-1]].neuron
                tmp_dx = barycentre[0] - 1
                tmp_dy = barycentre[1] - 1
                
                new_nrn3_curve["meta"]["curve"]["starting_point"]["x"] =  nrn2["meta"]["center"]["x"] + tmp_dx
                new_nrn3_curve["meta"]["curve"]["starting_point"]["y"] =  nrn2["meta"]["center"]["y"] + tmp_dy
                new_nrn3_curve["meta"]["curve"]["basis_vector"]["x"] -= tmp_dx
                new_nrn3_curve["meta"]["curve"]["basis_vector"]["y"] -= tmp_dy
                # normaliser le basis_vector
                new_nrn3_curve["meta"]["curve"]["basis_vector"]["x"] = new_nrn3_curve["meta"]["curve"]["basis_vector"]["x"] / np.sqrt(np.power(new_nrn3_curve["meta"]["curve"]["basis_vector"]["x"],2)+np.power(new_nrn3_curve["meta"]["curve"]["basis_vector"]["y"],2))
                new_nrn3_curve["meta"]["curve"]["basis_vector"]["y"] = new_nrn3_curve["meta"]["curve"]["basis_vector"]["y"] / np.sqrt(np.power(new_nrn3_curve["meta"]["curve"]["basis_vector"]["x"],2)+np.power(new_nrn3_curve["meta"]["curve"]["basis_vector"]["y"],2))
                print("###############################")
                print("nrn3", new_nrn3_curve["_id"])
                print("tmp_dx, tmp_dy", tmp_dx, tmp_dy)
                print("basis_vector", new_nrn3_curve["meta"]["curve"]["basis_vector"])
                # Ajouter la dbconnectivity
                for previous_nrn3_id in lst_nrn3_found:
                    previous_nrn3 = self.nrn_tls.lst_nrns[previous_nrn3_id-1].neuron
                    new_nrn3_curve["DbConnectivity"]["anti_lateral"].append(previous_nrn3["_id"])
                    previous_nrn3["DbConnectivity"]["lateral_connexion"].append(new_nrn3_curve["_id"])

                # le neurone ligne :
                lst_nrn3 = self.create_new_nrn3(lst_nrn3, nrn2, "sentive_vision_line")
                new_nrn3_2 = self.nrn_tls.lst_nrns[lst_nrn3[-1]].neuron
                print("###############################")
                print("nrn3", new_nrn3_2["_id"])

                # ajouter l'id du neurone courbe dans la liste des neurones connectés post-synaptique du neurone ligne
                new_nrn3_2["DbConnectivity"]["post_synaptique"].append(new_nrn3_curve["_id"])

                # list_coupled_nrn3s.append([new_nrn3_curve["_id"], new_nrn3_2["_id"]])

            elif nrn3_curve_not_found:
                # le neurone courbe :
                lst_nrn3 = self.create_new_nrn3(lst_nrn3, nrn2)
                new_nrn3 = self.nrn_tls.lst_nrns[lst_nrn3[-1]].neuron
                tmp_dx = barycentre[0] - 1
                tmp_dy = barycentre[1] - 1
                new_nrn3["meta"]["curve"]["starting_point"]["x"] =  nrn2["meta"]["center"]["x"] + tmp_dx
                new_nrn3["meta"]["curve"]["starting_point"]["y"] =  nrn2["meta"]["center"]["y"] + tmp_dy
                # new_nrn3["meta"]["curve"]["basis_vector"]["x"] += tmp_dx
                # new_nrn3["meta"]["curve"]["basis_vector"]["y"] += tmp_dy
                print("###############################")
                print("nrn3", new_nrn3["_id"])
                # Ajouter la dbconnectivity
                for previous_nrn3_id in lst_nrn3_found:
                    previous_nrn3 = self.nrn_tls.lst_nrns[previous_nrn3_id-1].neuron
                    new_nrn3["DbConnectivity"]["anti_lateral"].append(previous_nrn3["_id"])
                    previous_nrn3["DbConnectivity"]["lateral_connexion"].append(new_nrn3["_id"])

            # liste des neurones nrn3 trouvés
            lst_nrn3_found = copy.deepcopy(tmp_lst_nrn3_found)


            ##################################
            # détermination du pixel suivant :
            if len(new_pixels_detected) > 0:
                # rechercher s'il existe un max exclusif (c'est à dire un max qui n'est égal à aucun autre ?)
                max_iterations = 0
                nb_max_found = 0
                new_pixels_max = []
                for pixel_detected in new_pixels_detected:
                    if pixel_detected["nb_iteration"]>max_iterations:
                        max_iterations = pixel_detected["nb_iteration"]
                        nb_max_found = 1
                        new_pixels_max = [pixel_detected]
                    elif pixel_detected["nb_iteration"]==max_iterations:
                        nb_max_found += 1
                        new_pixels_max.append(pixel_detected)

                min_error = 999

                if nb_max_found == 1:
                    new_x = new_pixels_max[0]["x"]
                    new_y = new_pixels_max[0]["y"]
                elif nb_max_found >1:
                    for new_pixel in new_pixels_max:
                        tmp_error = np.abs(new_pixel["distance"] + new_pixel["angle"])
                        if tmp_error < min_error:
                            min_error = tmp_error
                            new_x = new_pixel["x"]
                            new_y = new_pixel["y"]
                            print("$$$> min_error", min_error, "[new_x:", new_x, ", new_y:", new_y, "]")

                print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
                print("Nombre de positions trouvées", len(new_pixels_detected))
                print("max_iterations", max_iterations)
                print("nb_max_found", nb_max_found)
                print("new_pixels_max", new_pixels_max)
                print("POSITION choisie:", new_x, new_y)
                print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        #                          _                 _             _          _    _             
        #    ____  _ _ __ _ __ _ _(_)_ __  ___ _ _  | |___ ___  __| |___ _  _| |__| |___ _ _  ___
        #   (_-< || | '_ \ '_ \ '_| | '  \/ -_) '_| | / -_|_-< / _` / _ \ || | '_ \ / _ \ ' \(_-<
        #   /__/\_,_| .__/ .__/_| |_|_|_|_\___|_|   |_\___/__/ \__,_\___/\_,_|_.__/_\___/_||_/__/
        #           |_|  |_|                                                                     
        # Faire une boucle sur tous les neurones curve et line pour supprimer les doublons
        # on défini un doublon par un neurone ayant le même starting_point
        # on supprime le doublon ayant le moins d'itérations
        # for coupled_nrn3 in list_coupled_nrn3s:
        #     nrn3_1 = self.nrn_tls.get_neuron_from_id(coupled_nrn3[0])
        #     nrn3_2 = self.nrn_tls.get_neuron_from_id(coupled_nrn3[1])
        #     if nrn3_1["meta"]["curve"]["nb_iteration"] >= nrn3_2["meta"]["line"]["nb_iteration"]:
        #         nrn3_to_remove = nrn3_2
        #     else:
        #         nrn3_to_remove = nrn3_1
        #     print("nrn3_to_remove", nrn3_to_remove["_id"])
        #     print("nrn3_to_remove", nrn3_to_remove)

        #     # on supprime le doublon
        #     self.nrn_tls.remove_nrn_by_id(nrn3_to_remove["_id"])


    def test_pos_predict_on_3x3_grid(self, nrn2, nrn3_pos_predict):
        if nrn3_pos_predict["x"] - nrn2["meta"]["center"]["x"] > 1:
            nrn3_pos_predict["x"] = nrn2["meta"]["center"]["x"] + 1
        elif nrn3_pos_predict["x"] - nrn2["meta"]["center"]["x"] < -1:
            nrn3_pos_predict["x"] = nrn2["meta"]["center"]["x"] - 1
        if nrn3_pos_predict["y"] - nrn2["meta"]["center"]["y"] > 1:
            nrn3_pos_predict["y"] = nrn2["meta"]["center"]["y"] + 1
        elif nrn3_pos_predict["y"] - nrn2["meta"]["center"]["y"] < -1:
            nrn3_pos_predict["y"] = nrn2["meta"]["center"]["y"] - 1
        return nrn3_pos_predict


    def create_new_nrn3(self, lst_nrn3, nrn2, nrn_type="sentive_vision_curve"):
        """_summary_
        Crée un nouveau neurone de la couche 3 et le connecte au neurone 2

        Args:
            lst_nrn3 (_type_): _description_
            nrn2 (_type_): _description_
            nrn_type (str, optional): _description_. Defaults to "sentive_vision_curve".
        """
            
        nb3  = self.nrn_tls.add_new_nrn(nrn_type)

        # on ajoute le neurone à la liste des neurones de la couche 2
        lst_nrn3.append(nb3)
        # racourci pour accéder au neurone
        nrn3 = self.nrn_tls.lst_nrns[nb3].neuron

        # rendre le vecteur orientation du neurone 2 unitaire
        # norm_vecteur = np.sqrt(np.power(nrn2["meta"]["orientation"]["x"],2)+np.power(nrn2["meta"]["orientation"]["y"],2))

        key_nrn_type = "curve"

        if nrn_type == "sentive_vision_curve":
            key_nrn_type = "curve"

        elif nrn_type == "sentive_vision_line":
            key_nrn_type = "line"

        # maj du starting_point c'est le centre du nrn2
        nrn3["meta"][key_nrn_type]["starting_point"] = nrn2["meta"]["center"]
        # maj du vecteur de base
        # nrn3["meta"][key_nrn_type]["basis_vector"]["x"] = np.sqrt(2) * nrn2["meta"]["orientation"]["x"]/norm_vecteur
        # nrn3["meta"][key_nrn_type]["basis_vector"]["y"] = np.sqrt(2) * nrn2["meta"]["orientation"]["y"]/norm_vecteur
        nrn3["meta"][key_nrn_type]["basis_vector"]["x"] = nrn2["meta"]["orientation"]["x"]
        nrn3["meta"][key_nrn_type]["basis_vector"]["y"] = nrn2["meta"]["orientation"]["y"]
        nrn3["meta"][key_nrn_type]["nb_iteration"] = 1
        nrn3["meta"]["pending_nb_iteration"] = 1

        # on connecte le nrn2 au nrn3
        nrn3["DbConnectivity"]["pre_synaptique"].append(nrn2["_id"])

        '''
        # on lui ajoute les pixels
        mtrx = np.zeros([28,28])
        # boucle sur les neurones pixels présynaptiques du neurone 2
        for nrn_pxl_id in nrn2["DbConnectivity"]["pre_synaptique"]:
            if nrn_pxl_id > 0:
                    # on récupère la matrice du neurone pixel
                mtrx = self.nrn_tls.get_neuron_receptive_field(nrn_pxl_id, mtrx)

            # on sauvegarde la matrice
        nrn3["meta"]["pixels_matrix"] = mtrx
        '''
        return lst_nrn3

    def find_tips(self, cp_lst_nrns, lthrshld_tip, lthrshld_nod, G, r_thrshld_tip=-1):
        l_tmp_tips = [] # id des neurones situés à une extrémité
        l_tmp_node = [] # id des neurones au carrefour
        l_tmp_stop = []
        liste_electeurs = []
        liste_candidats = []
        see_tips = []
        for pos in range(len(cp_lst_nrns)):
            # nrn = cp_lst_nrns[pos].neuron
            nrn = self.nrn_tls.get_neuron_from_id(cp_lst_nrns[pos])
            # Sélection des neurones TIPS
            see_tips.append(len(nrn["DbConnectivity"]["lateral_connexion"]))
            if len(nrn["DbConnectivity"]["lateral_connexion"])<=lthrshld_tip:
                l_tmp_tips.append(nrn["_id"])
            if nrn["ratio_conn"]<=r_thrshld_tip:
                l_tmp_tips.append(nrn["_id"])
            # Sélection des neurones NODES
            if len(nrn["DbConnectivity"]["lateral_connexion"])>=lthrshld_nod:
                l_tmp_stop.append(nrn["_id"])
                lbl_not_found = True
                for i in range(len(liste_electeurs)):
                    if len(set(liste_electeurs[i]).intersection(set(nrn["DbConnectivity"]["lateral_connexion"])))>0:
                        liste_electeurs[i].extend(nrn["DbConnectivity"]["lateral_connexion"])
                        liste_candidats[i].append(nrn["_id"])
                        lbl_not_found = False
                if lbl_not_found:
                    liste_electeurs.append(nrn["DbConnectivity"]["lateral_connexion"])
                    liste_candidats.append([nrn["_id"]])
        # print("seuil TIPS",lthrshld_tip)
        # print("see_tips\n",see_tips)
        # regroupement des nodes en trop grand nombres
        vainqueurs_elections = []
        # déroulement du scrutin
        for scrutin in range(len(liste_candidats)):
            liste_votants = list(set(liste_electeurs[scrutin]))
            resultats_election_locale = np.zeros(len(liste_candidats[scrutin]))
            # chaque électeur procède à son vote pour les candidats locaux liste_candidats[scrutin]
            for votant_id in range(len(liste_votants)):
                votant = self.nrn_tls.get_neuron_from_id(liste_votants[votant_id])
                if votant != '': 
                    vote = votant["DbConnectivity"]["weights"]
                    for candidat_id in range(len(liste_candidats[scrutin])):
                        try:
                            resultats_election_locale[candidat_id] += vote[liste_candidats[scrutin][candidat_id]]
                        except:
                            continue
            vainqueurs_elections.append(liste_candidats[scrutin][np.argmax(resultats_election_locale)])   
            # print("resultats_election_locale",resultats_election_locale,"candidats:",liste_candidats[scrutin],"vainqueur:",vainqueurs_elections[len(vainqueurs_elections)-1])
        l_tmp_node = vainqueurs_elections
        l_tmp_tips = list(set(l_tmp_tips))
        # longuest = []
        # for t in l_tmp_tips:
        #     tmp_path_length = []
        #     for n in l_tmp_node:
        #         tmp_path_length.append(int(nx.shortest_path_length(G,t,n)))
        #     longuest.append(min(tmp_path_length))
        # # réarrange les extrémités en fonction de la longueur des chemins les plus courts
        # l_tmp_tips = np.array(l_tmp_tips)
        # l_tmp_tips = l_tmp_tips[np.flip(np.argpartition(np.array(longuest),len(longuest)-1))]
        return l_tmp_tips, l_tmp_node, np.array(l_tmp_stop)

    
    def get_nrn_from_path(self, list_path_nrn_id):
        """A partir de la liste des neurones passés en paramètre,
            retourne l'ensemble des connexions latérales mobilisés
        Args:
            list_path_nrn_id (list): nrn_id
        """
        lst_output = []
        for nrn2_id in list_path_nrn_id:
            nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
            lst_output.extend(nrn2["DbConnectivity"]["lateral_connexion"])
        
        return set(lst_output)




    def new_angle_neuron(self, nrn2_1_id, nrn2_2_id, nrn2_3_id):
        # crée un nouveau neurone
        nb = self.nrn_tls.add_new_nrn("sentive_angle_neuron")
        nrn4 = self.nrn_tls.lst_nrns[nb].neuron
        nrn2_1 = self.nrn_tls.get_neuron_from_id(nrn2_1_id)
        nrn2_2 = self.nrn_tls.get_neuron_from_id(nrn2_2_id)
        nrn2_3 = self.nrn_tls.get_neuron_from_id(nrn2_3_id)
        self.nrn_tls.add_nrn_connexion(nrn4, nrn2_1)
        self.nrn_tls.add_nrn_connexion(nrn4, nrn2_2)
        self.nrn_tls.add_nrn_connexion(nrn4, nrn2_3)
        nrn4["meta"]["orientation"]["x"] = nrn2_3["meta"]["center"]["x"]-nrn2_1["meta"]["center"]["x"]
        nrn4["meta"]["orientation"]["y"] = nrn2_3["meta"]["center"]["y"]-nrn2_1["meta"]["center"]["y"]
        v1 = {"x":0,"y":0}
        v1["x"] = nrn2_2["meta"]["center"]["x"]-nrn2_1["meta"]["center"]["x"]
        v1["y"] = nrn2_2["meta"]["center"]["y"]-nrn2_1["meta"]["center"]["y"]
        v2 = {"x":0,"y":0}
        v2["x"] = nrn2_3["meta"]["center"]["x"]-nrn2_2["meta"]["center"]["x"]
        v2["y"] = nrn2_3["meta"]["center"]["y"]-nrn2_2["meta"]["center"]["y"]
        nrn4["meta"]["angle"] = self.nrn_tls.calc_angle(v1,v2)
        nrn4["meta"]["before_length"] = self.nrn_tls.calc_dist(nrn2_1["meta"]["center"], nrn2_2["meta"]["center"])
        nrn4["meta"]["after_length"] = self.nrn_tls.calc_dist(nrn2_3["meta"]["center"], nrn2_2["meta"]["center"])
    

    def run_layers(self):
        self.layer_1() # pixels
        self.layer_2() # triplets

        # self.show_layer_vectors(3, False)
        # self.calc_angles_layer_3()
        # self.show_vectors_directions(3, False)
        # self.plot_angles_3()


    def reset_episode(self):
        self.episode[:,:,0]=self.episode[:,:,1]


    def show_neuron_receptive_field(self, nrn_id, verbose=False):

        rcptv_fields = self.nrn_tls.get_neuron_receptive_field(nrn_id, self.episode[:,:,0], self.nrn_tls.lst_nrns, verbose)
        
        plt.matshow(rcptv_fields)
        self.reset_episode()


    def show_receptive_field(self, neuron_idx,):
        # Visualiser le champs récepteur du neurone
        current_neuron = self.nrn_tls.lst_nrns[neuron_idx].neuron
        sub_matrix = self.nrn_tls.get_neuron_receptive_field(current_neuron, self.episode)
        print(current_neuron)
        plt.matshow(sub_matrix)

    
    def show_all_fields(self,lint_width=-1):
        if lint_width ==-1:
            all_fields = self.nrn_tls.get_all_center_fields(self.lst_nrns, self.episode)
        else:
            all_fields = self.nrn_tls.get_all_center_fields_width(self.lst_nrns, self.episode,lint_width)
        # print(all_fields)
        plt.matshow(all_fields)
        self.reset_episode()


    def show_receptive_field_id(self, neuron_idx2):
        # Visualiser le champs récepteur du neurone
        # plt.matshow(np.zeros(np.shape(self.episode[:,:,0])))
        sub_matrix = self.nrn_tls.get_neuron_receptive_field(neuron_idx2, self.episode[:,:,0])
        # print(sub_matrix)
        plt.matshow(sub_matrix)


    def print_neurons_by_layer_id(self, layer_id):
        for item in self.nrn_tls.lst_nrns:
            if item.neuron["layer_id"]==layer_id:
                print(item.neuron["_id"],":",item.neuron["DbConnectivity"]["pre_synaptique"], item.neuron)

    
    def draw_path_from_nrns_id(self, path, ax=-1):
        if ax==-1:
            _, ax = plt.subplots()
        x_values =[]
        y_values = []
        for nrn2_id in path:
            nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
            if nrn2=='': continue
            x_values.append(nrn2["meta"]["center"]["x"])
            y_values.append(nrn2["meta"]["center"]["y"])
        ax.plot(x_values, y_values, "k+-")


    def show_nrn_path_by_id(self, nrn_id, ax=-1):
        nrn = self.nrn_tls.get_neuron_from_id(nrn_id)
        return self.draw_path_from_nrns_id(nrn["meta"]["path"], ax)


    def draw_binome_angle_by_id(self, pos_id, ax=-1):
        if ax==-1:
            _, ax = plt.subplots()
        ax.plot(self.nrn_saccade[pos_id]["angles"],"kx-")
        ax.plot(self.nrn_saccade[pos_id]["l_angles"],"r*--")
        # ax.plot(np.abs(self.nrn_saccade[pos_id]["joints"]),"g+--")
        # print("ecart-type",np.std(self.nrn_saccade[pos_id]["angles"]))
        print("ratio_pxls_total",self.nrn_saccade[pos_id]["ratio_pxls_total"])
        ax.grid(True, which='both')

        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')


    def draw_segment_path_by_pos(self, nrn_pos, ax=-1):
        return self.draw_path_from_nrns_id(self.nrn_segments[nrn_pos]["meta"]["path"], ax)

    
    def draw_binome_path_by_pos(self, nrn_pos, ax=-1):
        return self.draw_path_from_nrns_id(self.nrn_saccade[nrn_pos]["path"], ax)

    
    def draw_selected_segment_path(self, ax=-1):
        if ax==-1:
            _, ax = plt.subplots()
        print("SELECTED SEGMENTS",  self.slct_sgmts)
        for nrn_id in self.slct_sgmts:
            nrn = self.nrn_tls.get_segment_from_id(nrn_id,self.nrn_segments)
            if nrn != '':
                self.draw_path_from_nrns_id(nrn["meta"]["path"], ax)


    def show_selected_segment_pxl(self, pos=-1, ax=-1):
        if ax==-1:
            _, ax = plt.subplots()
        np_stamp = np.zeros([self.IMG_SIZE,self.IMG_SIZE])
        i = 0
        for nrn_id in self.slct_sgmts:
            # for nrn_pxl_id in nrn["nrn3"]["meta"]["mobilise_pxl_ids"]:
            if pos==-1 or pos==i:
                nrn = self.nrn_tls.get_segment_from_id(nrn_id,self.nrn_segments)
                # print (i, nrn)
                for nrn_pxl_id in nrn["nrn3"]["meta"]["mobilise_nrn2_ids"]:
                    nrnx = self.nrn_tls.get_neuron_from_id(nrn_pxl_id)
                    
                    if nrnx != '': 
                        np_stamp[nrnx["meta"]["center"]["y"],nrnx["meta"]["center"]["x"]]+=1
                for nrn_pth_id in nrn["meta"]["path"]:
                    nrnx = self.nrn_tls.get_neuron_from_id(nrn_pth_id)
                    if nrnx != '': 
                        np_stamp[nrnx["meta"]["center"]["y"],nrnx["meta"]["center"]["x"]]+=1
            if pos!=-1 and pos==i:
                break
            i+=1
        ax.matshow(np_stamp)

    def show_layer_vectors(self, layer_id, lbl_show_angles=True):
        X = []
        Y = []
        u_x = []
        u_y = []
        nb = 0
        for nrn in self.nrn_tls.lst_nrns:
            if nrn.neuron["layer_id"] == layer_id:
                nrn2 = nrn.neuron
                nb += 1
                X.append(nrn2["meta"]["glbl_prm"]["cg"]["x"])
                Y.append(nrn2["meta"]["glbl_prm"]["cg"]["y"])
                u_x.append(nrn2["meta"]["glbl_prm"]["u_axis"]["x"])
                u_y.append(nrn2["meta"]["glbl_prm"]["u_axis"]["y"])
        q = plt.quiver(X,Y,u_x,u_y)
        for nrn in self.nrn_tls.lst_nrns:
            if nrn.neuron["layer_id"] == layer_id:
                nrn2 = nrn.neuron
                x = nrn2["meta"]["glbl_prm"]["cg"]["x"]
                y = nrn2["meta"]["glbl_prm"]["cg"]["y"]
                plt.text(x,y, str(nrn2["_id"]))
                if not lbl_show_angles:
                    continue
                try:
                    angles = nrn2["DbConnectivity"]['angles']
                except:
                    continue
                for key in angles:
                    y += 0.5
                    angle = angles[key]
                    angle = np.abs(angle)
                    if angle > (np.pi)/2:
                        angle = np.pi - angle
                    if angle ==0:
                        color = "green"
                    elif angle < 0.1:
                        color = "yellow"
                    elif angle < 0.4:
                        color = "orange"
                    elif angle < 0.8:
                        color = "red"
                    else :
                        color = "purple"
                    plt.text(x,y, str(key),color=color)


    def calc_angles_layer_3(self, bool_remove_zeros = False):
        self.nrn_tls.new_layer()

        self.angle_mov_3 = []
        self.angles_3 = []
        self.ids_3 = []
        self.dist_3 = []

        self.angle_mov_4 = []
        self.angles_4 = []
        self.ids_4 = []
        self.dist_4 = []

        previous_nrn4_id = -1

        nrn3_list = {}
        crt_nrn3 = {}
        nrn3_nexts = {}
        lbl_1st_nrn3 = False
        for pool in self.nrn_tls.lst_nrns:
            if pool.neuron["layer_id"]==3:
                if not lbl_1st_nrn3:
                    crt_nrn3 = pool.neuron
                    lbl_1st_nrn3 = True
                else:
                    nrn3_list[pool.neuron["_id"]] = pool.neuron

        while (len(nrn3_list)):
            # tu vas charger dans next les suivants à partir de lateral_connexion
            for nxt_nrn_id in crt_nrn3["DbConnectivity"]["lateral_connexion"]:
                try:
                    nrn3_nexts[nxt_nrn_id] = nrn3_list.pop(nxt_nrn_id)
                except:
                    pass

            # pour chacun des suivants de nrn3_nexts tu vas noter l'angle avec le nrn 3 en cours : crt_nrn3
            min_angles = {}
            lbl_1st_nrn3 = False
            for tested_nrn3 in nrn3_nexts.values():
                try:
                    tmp_angle = tested_nrn3["DbConnectivity"]["angles"][crt_nrn3["_id"]]
                    if not lbl_1st_nrn3:
                        min_angles[tested_nrn3["_id"]] = tmp_angle
                        lbl_1st_nrn3 = True
                    elif np.abs(tmp_angle) < np.abs(list(min_angles.values())[0]):
                        min_angles = {}
                        min_angles[tested_nrn3["_id"]] = tmp_angle
                except:
                    pass

            try:
                # en fait tu as juste besoin de garder le minimum
                self.angles_3.append(list(min_angles.values())[0])
                self.ids_3.append(crt_nrn3["_id"])
                nxt_id = list(min_angles.keys())[0]
                cg1 = crt_nrn3["meta"]["glbl_prm"]["cg"]
                cg2 = nrn3_nexts[nxt_id]["meta"]["glbl_prm"]["cg"]
                try:
                    previous_vecteur_deplacement = crt_nrn3["meta"]["vecteur_deplacement"]
                except KeyError:
                    previous_vecteur_deplacement = crt_nrn3["meta"]["glbl_prm"]["u_axis"]
                new_vecteur_deplacement = {
                    "x": cg2["x"] - cg1["x"],
                    "y": cg2["y"] - cg1["y"],
                }
                self.angle_mov_3.append(self.nrn_tls.calc_angle(new_vecteur_deplacement, previous_vecteur_deplacement))
                # création d'un nouveau neurone 4
                nb = self.nrn_tls.add_new_nrn()
                nrn4 = self.nrn_tls.lst_nrns[nb].neuron
                nrn4["DbConnectivity"]["pre_synaptique"] = [crt_nrn3["_id"], nrn3_nexts[nxt_id]["_id"]]
                
                pxl_coords = set()
                # Pour chaque nrn 2 j'update la matrice des pixels
                try:
                    pxl_coords.update(set(nrn3_nexts[nxt_id]["meta"]["pxl_coord"]))
                    pxl_coords.update(set(crt_nrn3["meta"]["pxl_coord"]))
                except KeyError:
                    pass

                # ce qui me permet de calculer le vecteur directeur
                nrn4["meta"]["pxl_coord"] = list(pxl_coords)
                
                pca4 = PCA(n_components=1)
                pca4.fit(nrn4["meta"]["pxl_coord"])

                nrn4["meta"]["glbl_prm"] = {
                                                "cg":{  "x":np.mean(np.array(list(pxl_coords))[:,0]),
                                                        "y":np.mean(np.array(list(pxl_coords))[:,1])},
                                                "u_axis":{
                                                        "x":pca4.components_[0][0],
                                                        "y":pca4.components_[0][1]}
                                            }
                # Réorienter ici le u_axis
                # pour cela il faut le comparer au vecteur directeur
                vector_1 = nrn4["meta"]["glbl_prm"]["u_axis"]
                result_angle = np.abs(self.nrn_tls.calc_angle(vector_1, new_vecteur_deplacement))
                if result_angle>(np.pi/2):
                    nrn4["meta"]["glbl_prm"]["u_axis"]["x"] = - nrn4["meta"]["glbl_prm"]["u_axis"]["x"]
                    nrn4["meta"]["glbl_prm"]["u_axis"]["y"] = - nrn4["meta"]["glbl_prm"]["u_axis"]["y"]

                # ajouter les connexions latérales au nrn4
                if previous_nrn4_id != -1:
                    nrn4["DbConnectivity"]["lateral_connexion"].append(previous_nrn4_id)
                    prev_nrn4 = self.nrn_tls.get_neuron_from_id(previous_nrn4_id)
                    prev_nrn4["DbConnectivity"]["lateral_connexion"].append(nrn4["_id"])
                    # Ajoute ici l'angle avec le suivant dans le précédent
                    vector_2 = prev_nrn4["meta"]["glbl_prm"]["u_axis"]
                    self.angles_4.append(self.nrn_tls.calc_angle(vector_1, vector_2))
                    self.ids_4.append(nrn4["_id"])
                    new_vecteur_deplacement_4 = {
                        "x" : nrn4["meta"]["glbl_prm"]["cg"]["x"] - prev_nrn4["meta"]["glbl_prm"]["cg"]["x"],
                        "y" : nrn4["meta"]["glbl_prm"]["cg"]["y"] - prev_nrn4["meta"]["glbl_prm"]["cg"]["y"]
                    }
                    nrn4["meta"]["vecteur_deplacement"] = copy.deepcopy(new_vecteur_deplacement_4)
                    self.dist_4.append(np.sqrt(np.power(new_vecteur_deplacement_4["x"],2)+np.power(new_vecteur_deplacement_4["y"],2))) 

                    try:
                        vector_2 = prev_nrn4["meta"]["vecteur_deplacement"]
                        vector_1 = nrn4["meta"]["vecteur_deplacement"]
                        self.angle_mov_4.append(self.nrn_tls.calc_angle(vector_1, vector_2))
                    except KeyError:
                        self.angle_mov_4.append(0)


                    # Ajoute ici la distance avec le suivant dans le précédent
                
                
                previous_nrn4_id = nrn4["_id"]

                self.dist_3.append(np.sqrt(np.power(new_vecteur_deplacement["x"],2)+np.power(new_vecteur_deplacement["y"],2)))
                if nxt_id == 129:
                    print(nrn3_nexts[nxt_id]["meta"])
                nrn3_nexts[nxt_id]["meta"]["vecteur_deplacement"] = copy.deepcopy(new_vecteur_deplacement)
                if nxt_id == 100:
                    print(nrn3_nexts[nxt_id]["meta"])
                # L'heureux élu devient le neurone en court
                crt_nrn3 = nrn3_nexts.pop(nxt_id)
                # tous ceux qui restent dans la liste des suivants sont remis dans la liste principale
                for tested_nrn3 in nrn3_nexts.values():
                    nrn3_list[tested_nrn3["_id"]] = tested_nrn3
            except:
                break
        
        # La distance est la distance cumulée et pas la distance entre chaque point
        self.x_dist = np.cumsum(self.dist_3).tolist()
        self.x_dist_4 = np.cumsum(self.dist_4).tolist()

        if bool_remove_zeros:
            for i in range (len(self.angles_3)-1, 0, -1):
                if self.angles_3[i] == 0:
                    self.angles_3.pop(i)
                    self.x_dist.pop(i)
                    self.ids_3.pop(i)
    
    def plot_angles_3(self):
        _, ax = plt.subplots()
        ax.plot(self.x_dist,self.angles_3)
        x = 0
        for id in self.ids_3:
            y = self.angles_3[x]
            z = self.x_dist[x]
            ax.text(z,y, str(id))
            x += 1


    def plot_angles_4(self):
        _, ax = plt.subplots()
        ax.plot(self.x_dist_4,self.angles_4)
        x = 0
        for id in self.ids_4:
            y = self.angles_4[x]
            z = self.x_dist_4[x]
            ax.text(z,y, str(id))
            x += 1


    def plot_angles_mov_3(self):
        _, ax = plt.subplots()
        ax.plot(self.x_dist,self.angle_mov_3)
        x = 0
        for id in self.ids_3:
            y = self.angle_mov_3[x]
            z = self.x_dist[x]
            ax.text(z,y, str(id))
            x += 1


    def plot_angles_mov_4(self):
        _, ax = plt.subplots()
        ax.plot(self.x_dist_4,self.angle_mov_4)
        x = 0
        for id in self.ids_4:
            y = self.angle_mov_4[x]
            z = self.x_dist_4[x]
            ax.text(z,y, str(id))
            x += 1


    def show_vectors_directions(self, layer_id, lbl_show_angles=True):
        X = []
        Y = []
        u_x = []
        u_y = []
        nb = 0
        for nrn in self.nrn_tls.lst_nrns:
            if nrn.neuron["layer_id"] == layer_id:
                nrn2 = nrn.neuron
                try:
                    u_x.append(nrn2["meta"]["vecteur_deplacement"]["x"])
                    u_y.append(nrn2["meta"]["vecteur_deplacement"]["y"])
                    X.append(nrn2["meta"]["glbl_prm"]["cg"]["x"])
                    Y.append(nrn2["meta"]["glbl_prm"]["cg"]["y"])
                    nb += 1
                except KeyError:
                    u_x.append(nrn2["meta"]["glbl_prm"]["u_axis"]["x"])
                    u_y.append(nrn2["meta"]["glbl_prm"]["u_axis"]["y"])
                    X.append(nrn2["meta"]["glbl_prm"]["cg"]["x"])
                    Y.append(nrn2["meta"]["glbl_prm"]["cg"]["y"])
                    nb += 1

        _, ax = plt.subplots()     
        ax.quiver(X,Y,u_x,u_y)
        for nrn in self.nrn_tls.lst_nrns:
            if nrn.neuron["layer_id"] == layer_id:
                nrn2 = nrn.neuron
                x = nrn2["meta"]["glbl_prm"]["cg"]["x"]
                y = nrn2["meta"]["glbl_prm"]["cg"]["y"]
                ax.text(x,y, str(nrn2["_id"]))
                if not lbl_show_angles:
                    continue
                try:
                    angles = nrn2["DbConnectivity"]['angles']
                except:
                    continue
                for key in angles:
                    y += 0.5
                    angle = angles[key]
                    angle = np.abs(angle)
                    if angle > (np.pi)/2:
                        angle = np.pi - angle
                    if angle ==0:
                        color = "green"
                    elif angle < 0.1:
                        color = "yellow"
                    elif angle < 0.4:
                        color = "orange"
                    elif angle < 0.8:
                        color = "red"
                    else :
                        color = "purple"
                    ax.text(x,y, str(key),color=color)


    def show_vectors_directions_nrn3_arc(self):
        X = []
        Y = []
        b_x = []
        b_y = []
        v_x = []
        v_y = []
        nb = 0
        _, ax = plt.subplots()
        for nrn in self.nrn_tls.lst_nrns:
            if nrn.neuron["layer_id"] == 3:
                nrn3 = nrn.neuron
                v_x.append(nrn3["meta"]["last_nrn2"]["vecteur_deplacement"]["x"])
                v_y.append(nrn3["meta"]["last_nrn2"]["vecteur_deplacement"]["y"])
                b_x.append(nrn3["meta"]["curve"]["basis_vector"]["x"])
                b_y.append(nrn3["meta"]["curve"]["basis_vector"]["y"])
                x = nrn3["meta"]["last_nrn2"]["glbl_prm"]["cg"]["x"]
                X.append(x)
                y = nrn3["meta"]["last_nrn2"]["glbl_prm"]["cg"]["y"]
                Y.append(y)
                # afficher les id des neurones et l'angle de rotation
                txt_to_show = f"{nrn3['_id']}, a:{np.round(nrn3['meta']['curve']['rotation_angle'],2)}({np.round(nrn3['meta']['curve']['angle_v_deplacement'],2)})"
                ax.text(x,y, txt_to_show)
        # afficher les vecteurs sur graphique
        ax.quiver(X,Y,v_x,v_y,color="red")
        print("vecteur déplacement en rouge")
        ax.quiver(X,Y,b_x,b_y)


    def show_graph_angle_f_distance_nrn3(self):
        _, ax = plt.subplots()
        x_dist = []
        angles_3 = []
        for nrn in self.nrn_tls.lst_nrns:
            if nrn.neuron["layer_id"] == 3:
                nrn3 = nrn.neuron
                x_dist.append(nrn3["meta"]["curve"]["distance"])
                angles_3.append(nrn3["meta"]["curve"]["angle_v_deplacement"])

        ax.plot(x_dist,angles_3)
            # x = 0
            # for id in self.ids_3:
            #     y = self.angles_3[x]
            #     z = self.x_dist[x]
            #     ax.text(z,y, str(id))
            #     x += 1