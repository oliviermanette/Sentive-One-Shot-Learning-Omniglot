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


    def layer_2_v2(self):
        ##################################################
        ########## NEURONES DE LA COUCHE 2 (t_3) #########
        ##################################################
        # Les neurones de cette couche ont des champs récepteurs 
        # qui sont des matrices de *3x3* mais orientés

        # copie locale de nrn_pxl_map
        nrn_pxl_map = copy.deepcopy(self.nrn_pxl_map)
        
        # Création de la nouvelle couche
        self.nrn_tls.new_layer()
        # liste contenant les id de la couche 2
        lst_nrn2_pos = []

        # on crée un premier neurone de la couche 2
        nb  = self.nrn_tls.add_new_nrn()
        # id minimum des neurones de la couche 2
        nb_min = nb
        # on ajoute le neurone à la liste des neurones de la couche 2
        lst_nrn2_pos.append(nb)
        # racourci pour accéder au neurone
        nrn2 = self.nrn_tls.lst_nrns[nb].neuron

        # on modifie les paramètres du neurone créé
        x = self.nrn_tls.lst_nrns[0].neuron["meta"]["center"]["x"]
        nrn2["meta"]["center"]["x"] = x
        y = self.nrn_tls.lst_nrns[0].neuron["meta"]["center"]["y"]
        nrn2["meta"]["center"]["y"] = y
        nrn2["meta"]["matrix_width"] = 3

        # pixel central du neurone
        central_pixel_id = nrn_pxl_map[y][x]

        # supprime le pixel central de la map
        nrn_pxl_map[y][x] = 0

        # sub_pxl_map contient les identifiants de chaque neurone pixel sur une carte nrnl_map
        sub_pxl_map = nrn_pxl_map[y-1:y+2, x-1:x+2]

        nrn2["meta"]["sub_pxl_map"] = sub_pxl_map

        # ajoute les id des neurones pixels dans la liste des pre_synaptique
        tmp_list_sub_pxl = list(set(sub_pxl_map.ravel()))
        nrn2["DbConnectivity"]["pre_synaptique"] = tmp_list_sub_pxl
        # ajoute le neurone central dans la liste des pre_synaptique
        nrn2["DbConnectivity"]["pre_synaptique"].append(central_pixel_id)

        # supprime les neurones pixels dans nrn_pxl_map qui sont dans le champ récepteur du neurone
        for tmp_y in range(y-1,y+2):
            for tmp_x in range(x-1,x+2):
                nrn_pxl_map[tmp_y][tmp_x] = 0

        ## calcule le vecteur d'orientation moyen des pixels
        # pour chaque neurone pixel de la liste sub_pxl_map on calcule le vecteur d'orientation
        x_composant = []
        y_composant = []
        weight_sum = 0
        for nrn_pxl_id in tmp_list_sub_pxl:
            if nrn_pxl_id > 0:
                # on récupère le neurone pixel
                nrn_pxl = self.nrn_tls.lst_nrns[nrn_pxl_id-1].neuron
                # on pondère par le poids du neurone pixel
                weight_sum += nrn_pxl["weight"]
                # on récupère le vecteur d'orientation du neurone pixel
                x_composant.append(nrn_pxl["weight"]*(nrn_pxl["meta"]["center"]["x"] - nrn2["meta"]["center"]["x"]))
                y_composant.append(nrn_pxl["weight"]*(nrn_pxl["meta"]["center"]["y"] - nrn2["meta"]["center"]["y"]))

        # on fait la moyenne des composantes
        x_composant = np.mean(x_composant)/weight_sum
        y_composant = np.mean(y_composant)/weight_sum






    
    def layer_2(self):
        ##################################################
        ########## NEURONES DE LA COUCHE 2 (t_3) #########
        ##################################################
        # Les neurones de cette couche ont des champs récepteurs 
        # qui sont des matrices de *3x3*
        # avec des mata paramètres les décrivants.
        self.nrn_tls.new_layer()
        
        lst_nrn2_pos = []

        nb_min = 0

        for neuron_idx in range(self.nrn_tls.nb_nrns):
            # position du centre du neurone
            x = self.nrn_tls.lst_nrns[neuron_idx].neuron["meta"]["center"]["x"]
            y = self.nrn_tls.lst_nrns[neuron_idx].neuron["meta"]["center"]["y"]

            # sub_pxl_map contient les identifiants de chaque neurone pixel sur une carte nrnl_map
            sub_pxl_map = self.nrn_pxl_map[y-1:y+2, x-1:x+2]

            # crée un nouveau neurone de taille 3
            nb  = self.nrn_tls.add_new_nrn()
            if nb_min == 0:
                nb_min = nb
            lst_nrn2_pos.append(nb)
            nrn2 = self.nrn_tls.lst_nrns[nb].neuron
            if nrn2["_id"] == 61:
                print("sub_pxl_map")
            nrn2["meta"]["center"]["x"] = x
            nrn2["meta"]["center"]["y"] = y
            nrn2["meta"]["matrix_width"] = 3
            nrn2["meta"]["sub_pxl_map"] = sub_pxl_map
            nrn2["DbConnectivity"]["pre_synaptique"] = list(set(sub_pxl_map.ravel()))
            nrn2["meta"]["pxl_coord"] = []
            nrn2["meta"]["glbl_prm"] = {
                                            "cg":{"x":0,"y":0},
                                            "u_axis":{"x":0,"y":0}
                                        }

            self.nrn_l2_map[y][x] = nrn2["_id"]

            for i in range(len(nrn2["DbConnectivity"]["pre_synaptique"])-1,-1,-1):
                if nrn2["DbConnectivity"]["pre_synaptique"][i]==0:
                    nrn2["DbConnectivity"]["pre_synaptique"].pop(i)
                else:
                    self.nrn_tls.netGraph.add_edge(nrn2["_id"],nrn2["DbConnectivity"]["pre_synaptique"][i])
                    nrn_pxl = self.nrn_tls.get_neuron_from_id(nrn2["DbConnectivity"]["pre_synaptique"][i])
                    nrn_pxl["DbConnectivity"]["post_synaptique"].append(nrn2["_id"])

                    # ici tu dois récupérer les coordonnées du neurone présynaptique central -> nrn_pxl["meta"]["center"]["x"]
                    x = nrn_pxl["meta"]["center"]["x"]
                    y = nrn_pxl["meta"]["center"]["y"]
                    nrn2["meta"]["pxl_coord"].append((x,y))

            # calcul du PCA permettant d'obtenir l'orientation globale des pixels
            pca = PCA(n_components=1)
            pca.fit(nrn2["meta"]["pxl_coord"])
            # on obtient les résultats ici:
            # print(pca.components_)
            # permet d'avoir l'orientation globale du caractère
            nrn2["meta"]["glbl_prm"]["u_axis"]["x"]=pca.components_[0][0]
            nrn2["meta"]["glbl_prm"]["u_axis"]["y"]=pca.components_[0][1]

            # calcule le centre de gravité des pixels
            self.np_coord = np.array(nrn2["meta"]["pxl_coord"])
            nrn2["meta"]["glbl_prm"]["cg"]["x"] = np.mean(self.np_coord[:,0])
            nrn2["meta"]["glbl_prm"]["cg"]["y"] = np.mean(self.np_coord[:,1])
                    

            # print("neurone",nb,"list pre_synaptique")
            # print(self.nrn_tls.lst_nrns[nb].neuron["DbConnectivity"]["pre_synaptique"])

        # détermination des connexions latérales
        for nrn_pos in range (nb_min, self.nrn_tls.nb_nrns):
            nrn2 = self.nrn_tls.lst_nrns[nrn_pos].neuron
            # position du centre du neurone
            x = nrn2["meta"]["center"]["x"]
            y = nrn2["meta"]["center"]["y"]

            # sub_pxl_map contient les identifiants de chaque neurone pixel sur une carte nrnl_map
            sub_pxl_map = self.nrn_l2_map[y-1:y+2, x-1:x+2]
            nrn2["DbConnectivity"]["lateral_connexion"] = list(set(sub_pxl_map.ravel().astype(int)))
            for i_pos in range(len(nrn2["DbConnectivity"]["lateral_connexion"])-1,-1,-1):
                if nrn2["DbConnectivity"]["lateral_connexion"][i_pos] == 0 :
                    nrn2["DbConnectivity"]["lateral_connexion"].pop(i_pos)
                else:
                    self.nrn_tls.add_edge(nrn2["_id"],nrn2["DbConnectivity"]["lateral_connexion"][i_pos])
                    self.nrn_tls.increment_weight(nrn2,nrn2["DbConnectivity"]["lateral_connexion"][i_pos])

            # print("neurone",nrn2["_id"],"list lateral_connexion")
            # print(nrn2["DbConnectivity"]["lateral_connexion"])
            # Ajouter ici le calcul des angles avec chaque
            nrn2["DbConnectivity"]["angles"] = {}
            vector_1 = nrn2["meta"]["glbl_prm"]["u_axis"]
            for i_pos in range(len(nrn2["DbConnectivity"]["lateral_connexion"])-1,-1,-1):
                id_nrn = nrn2["DbConnectivity"]["lateral_connexion"][i_pos]
                if nrn2["_id"] != id_nrn:
                    nrn_2 = self.nrn_tls.get_neuron_from_id(id_nrn)
                    vector_2 = nrn_2["meta"]["glbl_prm"]["u_axis"]
                    nrn2["DbConnectivity"]["angles"][id_nrn] = self.nrn_tls.calc_angle(vector_1, vector_2)


        self.nrn_tls.nb_2_1st_layers = len(self.nrn_tls.lst_nrns)
        print("nb neurones couche 2 :", nb - nb_min)
        print("nombre de neurones couche 1 & 2:",self.nrn_tls.nb_2_1st_layers)
        # print("*"*40)

    def couche_3(self):
        '''
            Cette couche détermine les angles de rotation du trait en chaque point.
        '''
        # création d'une nouvelle couche
        self.nrn_tls.new_layer()

        #création d'un nouveau neurone
        nb  = self.nrn_tls.add_new_nrn("sentive_arc_neuron")
        self.nb_min_nrn3 = nb
        lst_nrn3_pos = [nb]
        lst_nrn3 = {}
        nrn3 = self.nrn_tls.lst_nrns[nb].neuron
        lst_nrn3[nrn3['_id']] = nrn3
        prevs_nrn3 = [nrn3]

        # initialisation du panier contenant la liste de tous les neurones 2 restants
        remaining_nrn2_id = {}
        i = 0
        # Pour cela je fais une boucle sur tous les nrn2 et je les copie dans mon panier
        for nrn in self.nrn_tls.lst_nrns:
            if nrn.neuron["layer_id"] == 2:
                nrn2 = nrn.neuron
                if i==0: 
                    crnt_nrn = copy.deepcopy(nrn2)
                else:
                    remaining_nrn2_id[nrn2["_id"]] = nrn2
                i += 1
        # Le premier nrn3 est par défaut connecté au premier nrn2 de la liste
        nrn3["DbConnectivity"]["pre_synaptique"].append(crnt_nrn["_id"])
        # Les nrn2 possèdent des "glbl_prm" qui donnent le cg des pixels et l'orientation.
        nrn3["meta"]["last_nrn2"] = {
            "glbl_prm" : crnt_nrn["meta"]["glbl_prm"],
            "vecteur_deplacement" : np.nan
        }
        max_angle = 0

        # charge la liste des précédents par défaut
        all_prevs = [[crnt_nrn]]

        crt_branch = 0
        # Tant qu'il existe des précédent on peut potentiellement continuer
        while len(all_prevs[crt_branch])>0:
            nexts_list = {}
            # calcule les suivant à partir de la liste DbConnectivity.angles
            # puisque cette liste donne tous les neurones avec qui ce nrn2 est connecté
            for nrn in all_prevs[crt_branch]:
                for nrn_id in nrn["DbConnectivity"]["angles"].keys():
                    try:
                        nexts_list[nrn_id] = remaining_nrn2_id[nrn_id]
                        remaining_nrn2_id.pop(nrn_id)
                    except:
                        pass
            # Donc on obtient une liste de tous les suivants possibles à partir des neurones contenus dans all_prevs
            # si cette liste est vide alors on arrête
            if len(nexts_list)==0:
                print("nothing more to do")
                break
            # Variables permettant de stocker les différentes branches
            branchs = []
            branchs.append({})
            # on sélectionne le 1er neurone de la liste des suivants, il devient le *neurone en court* `crn_nrn`
            crn_nrn = list(nexts_list.values())[0]
            # je charge ce neurone en court comme premier de ma première branche
            branchs[0][crn_nrn["_id"]] = crn_nrn
            prev_id = crn_nrn["_id"]

            # variable qui stocke quelle branche on se trouve. C'est simplement un chiffre entier de 0 à n
            crn_brnch_id = 0
            
            # Comme on a chargé le prochain neurone, il faut le supprimer de la liste des suivant pour les prochaines fois
            nexts_list.pop(crn_nrn["_id"])
            # On boucle sur chaque suivants de la liste
            while len(nexts_list)>0:
                # On met dans la même branche tous les neurones connectés entre eux
                for nrn_id in crn_nrn["DbConnectivity"]["angles"].keys():
                    try:
                        # ici avant d'ajouter mon neurone à la branche je vérifie son angle

                        tmp_angle = np.abs(nexts_list[nrn_id]["DbConnectivity"]["angles"][prev_id])
                        if tmp_angle > np.pi/2:
                            tmp_angle = np.pi - tmp_angle
                        if tmp_angle>= 1:
                            print("**************************************************")
                            print("angle avec le neurone déjà présent dans la branche",prev_id,"et actuel id:",nrn_id)
                            print("angle brut = ", nexts_list[nrn_id]["DbConnectivity"]["angles"][prev_id])
                            print("angle ajusté = ",tmp_angle)
                            print("**************************************************")
                            #continue
                        else:
                            branchs[crn_brnch_id][nrn_id] = nexts_list[nrn_id]
                            nexts_list.pop(nrn_id)
                    except:
                        #print("nothing to be done")
                        pass
                # Si à la fin du premier passage il reste des neurones
                # ça veut dire que ces neurones ne sont pas connectés avec les autres d'avant
                # on crée donc une seconde branche pour eux
                if len(nexts_list)>0:
                    crn_brnch_id += 1
                    branchs.append({})
                    crn_nrn = list(nexts_list.values())[0]
                    prev_id = crn_nrn["_id"]
                    branchs[crn_brnch_id][crn_nrn["_id"]] = crn_nrn
                    nexts_list.pop(crn_nrn["_id"])

                # et ainsi de suite tant qu'il ne reste plus de neurones dans la liste des suivants.
                # on aura créé une variable **branchs** qui contient des neurones de la 2 eme couche organisés en différentes branches.

            # Cette variable va stocker tous les neurones de la 3eme couche créés maintenant
            # cela va permettre de savoir qui sont les neurones 3 du niveau précédent quand on sera au niveau suivant
            # pour l'instant c'est juste un tableau qui permet de stocker ces neurones créés juste après
            new_prevs_nrn3 = []
            
            # réinitialise les branchs
            all_prevs=[]
            all_prev_nrn2s = []
            # print("number of branches is", len(branchs))
            for i in range(len(branchs)):
                all_prevs.append([])
                all_prev_nrn2s.append([])                   

            # boucle sur chaque branche il y trouve une structure qui contient tous les neurones 2 de la branche
            for strc_nrn2 in branchs:

                # boucle sur tous les neurones 3 déjà créés
                # pour voir si un de ces neurones 3 est compatible avec les neurones 2 de la branche.
                for a_nrn3 in lst_nrn3.values():
                    # comparaison des vecteurs d'orientation
                    for nrn2 in strc_nrn2.values():
                        # vecteur d'orientation du nrn2:
                        vector_1 = nrn2["meta"]["glbl_prm"]["u_axis"]
                        # vecteur d'orientation du nrn3
                        vector_2 = a_nrn3["meta"]["last_nrn2"]["glbl_prm"]["u_axis"]
                        result_angle = np.abs(self.nrn_tls.calc_angle(vector_1, vector_2))
                        if result_angle>(np.pi/2): 
                            result_angle -= np.pi/2

                    pass
                
                # Création d'un nouveau neurone 3 qui va être connecté à tous les neurones de la branche
                nb  = self.nrn_tls.add_new_nrn()
                lst_nrn3_pos.append(nb)
                nrn3 = self.nrn_tls.lst_nrns[nb].neuron
                nrn3["DbConnectivity"]["pre_synaptique"].extend(list(strc_nrn2.keys()))

                pxl_coords = set()
                # Pour chaque nrn 2 j'update la matrice des pixels
                for nrn2 in strc_nrn2.values():
                    pxl_coords.update(set(nrn2["meta"]["pxl_coord"]))
                    all_prev_nrn2s[crt_branch].append(nrn2["_id"])

                # ce qui me permet de calculer le vecteur directeur
                nrn3["meta"]["pxl_coord"] = list(pxl_coords)
                pca = PCA(n_components=1)
                pca.fit(nrn3["meta"]["pxl_coord"])

                nrn3["meta"]["glbl_prm"] = {
                                                "cg":{  "x":np.mean(np.array(list(pxl_coords))[:,0]),
                                                        "y":np.mean(np.array(list(pxl_coords))[:,1])},
                                                "u_axis":{
                                                        "x":pca.components_[0][0],
                                                        "y":pca.components_[0][1]}
                                            }
                nrn3["meta"]["vecteur_deplacement"] = {
                                                        "x": nrn3["meta"]["glbl_prm"]["cg"]["x"] - prevs_nrn3[0]["meta"]["glbl_prm"]["cg"]["x"],
                                                        "y": nrn3["meta"]["glbl_prm"]["cg"]["y"] - prevs_nrn3[0]["meta"]["glbl_prm"]["cg"]["y"]
                                                    }
                vector_1 = nrn3["meta"]["glbl_prm"]["u_axis"]
                vector_2 = nrn3["meta"]["vecteur_deplacement"]
                result_angle = np.abs(self.nrn_tls.calc_angle(vector_1, vector_2))
                if result_angle>(np.pi/2):
                    nrn3["meta"]["glbl_prm"]["u_axis"]["x"] = - nrn3["meta"]["glbl_prm"]["u_axis"]["x"]
                    nrn3["meta"]["glbl_prm"]["u_axis"]["y"] = - nrn3["meta"]["glbl_prm"]["u_axis"]["y"]

                # Check if the previous nrn3 has a vector in the right direction

                if not "vecteur_deplacement" in prevs_nrn3[0]["meta"]:
                    vector_1 = prevs_nrn3[0]["meta"]["glbl_prm"]["u_axis"]
                    result_angle = np.abs(self.nrn_tls.calc_angle(vector_1, vector_2))
                    if result_angle>(np.pi/2):
                        prevs_nrn3[0]["meta"]["glbl_prm"]["u_axis"]["x"] = - prevs_nrn3[0]["meta"]["glbl_prm"]["u_axis"]["x"]
                        prevs_nrn3[0]["meta"]["glbl_prm"]["u_axis"]["y"] = - prevs_nrn3[0]["meta"]["glbl_prm"]["u_axis"]["y"]
                
                nrn3["meta"]["angles"] = {}
                nrn3["DbConnectivity"]["angles"] = {}

                all_prevs[crt_branch].extend(list(strc_nrn2.values()))
                new_prevs_nrn3.append(nrn3)
                vector_1 = nrn3["meta"]["glbl_prm"]["u_axis"]

                for prev_nrn3 in prevs_nrn3:
                    # print("prev_nrn3",prev_nrn3)
                    prev_nrn3["DbConnectivity"]["lateral_connexion"].append(nrn3["_id"])
                    nrn3["DbConnectivity"]["lateral_connexion"].append(prev_nrn3["_id"])
                    vector_2 = prev_nrn3["meta"]["glbl_prm"]["u_axis"]
                    nrn3["DbConnectivity"]["angles"][prev_nrn3["_id"]] = self.nrn_tls.calc_angle(vector_1, vector_2)
                    try:
                        prev_nrn3["DbConnectivity"]["angles"][nrn3["_id"]] = nrn3["DbConnectivity"]["angles"][prev_nrn3["_id"]]
                    except:
                        pass
                crt_branch += 1

            prevs_nrn3 = new_prevs_nrn3

            crt_branch = 0
            if len(all_prevs[crt_branch])==0:
                try:
                    all_prevs = [[remaining_nrn2_id.pop(list(remaining_nrn2_id.keys())[0])]]
                except:
                    all_prevs = [[]]

        print(max_angle)


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


    def layer_3(self):
        """ Détermine * les neurones séquences *
        """
        # création d'une nouvelle couche 
        self.nrn_tls.new_layer()
        
        # lien vers la 2eme couche:
        G = self.nrn_tls.layer_graph[1]
        # G = self.nrn_tls.netGraph
        G2 = copy.deepcopy(G)

        pos_2_cut_off = {
                "connector":[],
                "position":[],
                "nrn_id":[],
                "crnt_nrn":[]
            }

        bl_got_segmented = False

        # copie de la liste des pixels
        # cp_lst_pxls = copy.deepcopy(self.nrn_pxls)

        cp_lst_nrns = [] # copie de la liste des neurones de la 2eme couche
        # calcul des ratios pour chaque neurone
        nrn_ratio_conn = []
        nb_ltrl_conn = []
        for nrn_pos in self.nrn_tls.lst_nrns:
            nrn = nrn_pos.neuron
            if nrn["layer_id"]==2:
                cp_lst_nrns.append(nrn["_id"])
                nb_ltrl_conn.append(len(nrn["DbConnectivity"]["lateral_connexion"]))
                tmp_num_conn = []                
                for nrn_lat in nrn["DbConnectivity"]["lateral_connexion"]:
                    tmp_num_conn.append(len(self.nrn_tls.lst_nrns[nrn_lat-1].neuron["DbConnectivity"]["lateral_connexion"]))
                new_ratio = len(nrn["DbConnectivity"]["lateral_connexion"])/np.mean(tmp_num_conn)
                nrn["ratio_conn"] = new_ratio
                nrn_ratio_conn.append(new_ratio)

        # seuil nombre relatif de connexions
        # r_thrshld_tip = min(nrn_ratio_conn)
        lthrshld_tip = min(nb_ltrl_conn)+1 # seuil pour détecter les extrémités
        lthrshld_nod = max(nb_ltrl_conn) # seuil pour détecter les nœuds


        # Sélectionner plusieurs tips potentiels dans la liste.
        l_tmp_tips, nrn_nodes_id, nrn_stop_id = self.find_tips(cp_lst_nrns, lthrshld_tip, lthrshld_nod, G)
        # print("neurons TIPS:",l_tmp_tips,", nrn NODES",nrn_nodes_id, "et neurons STOP:",nrn_stop_id)
        # initial_tips = copy.deepcopy(l_tmp_tips)
        if len(l_tmp_tips)<=1:
            l_tmp_tips.extend(nrn_stop_id)

        nb_max = len(cp_lst_nrns)
        int_limit = 10
        reste_percent = 100
        nb_min = -1
        print("taille neurones à séquencer :", nb_max)

        while reste_percent>=self.MIN_PATH and int_limit>=0:
            # Calculer les distances neuronales entre chacun
            tip_max_length = 0
            tip_1 = -1
            tip_2 = -1
            for pos_tp_1 in range(len(l_tmp_tips)-1):
                for pos_tp_2 in range(pos_tp_1+1, len(l_tmp_tips)):
                    try:
                        tmp_max_length = nx.shortest_path_length(G2, source=l_tmp_tips[pos_tp_1], target=l_tmp_tips[pos_tp_2])
                    except:
                        tmp_max_length = 0
                    if tip_max_length<tmp_max_length:
                        tip_max_length = tmp_max_length
                        tip_1 = l_tmp_tips[pos_tp_1]
                        tip_2 = l_tmp_tips[pos_tp_2]

            # Sélectionner celui qui a la distance la plus longue.
            # en faire le chemin 
            if tip_1!=-1 and tip_2!=-1:
                first_path = nx.shortest_path(G2, source=tip_1, target=tip_2)
                nrn_activated = self.get_nrn_from_path(first_path)

                # Supprime la liste des neurones:
                for nrn_id in nrn_activated:
                    try:
                        G2.remove_node(nrn_id)
                    except:
                        pass
                        # print("The node",nrn_id,"is not in the graph.")
                # Récupère la liste des neurones mobilisés et fait la différence ''
                # print("cp_lst_nrns\n", cp_lst_nrns)
                cp_lst_nrns = list(set(cp_lst_nrns).difference(nrn_activated))
                l_tmp_tips = cp_lst_nrns

                nb = self.nrn_tls.add_new_nrn("sentive_sequence_nrn")
                nrn3 = self.nrn_tls.lst_nrns[nb].neuron
                self.slct_sgmts.append(nrn3["_id"])
                nrn3["meta"]["path"] = first_path
                nrn3["meta"]["mobilise_nrn2_ids"] = nrn_activated
                nrn3["DbConnectivity"]["pre_synaptique"] = nrn_activated
                nrn3["meta"]["ratio_pxls_total"] = len(nrn_activated)/self.nb_nrn_pxls
                self.nrn_segments.append(nrn3)
                # print("nouveau neurone 3 créé:", nb,", path:", first_path)
                print("taille neurones à séquencer :", len(cp_lst_nrns))

                if nb_min ==-1:
                    nb_min = nb
                reste_percent = 100*len(cp_lst_nrns)/nb_max
            else:
                print("cannot create any layer_3 neuron", tmp_max_length)

            int_limit -= 1

        # print
        segmented_path = self.nrn_tls.lst_nrns[nb_min].neuron["meta"]["path"]
        
        for nrn_pos in range(nb_min+1, len(self.nrn_tls.lst_nrns)):
            crnt_nrn = self.nrn_tls.lst_nrns[nrn_pos].neuron
            path_crnt = crnt_nrn["meta"]["path"]
            path_f1rst = self.nrn_tls.lst_nrns[nb_min].neuron["meta"]["mobilise_nrn2_ids"]
            # print("path_f1rst",path_f1rst)
            tip_1 = path_crnt[0]
            # récupérer les neurones connectés au tip_1
            candidates = self.get_nrn_from_path([tip_1])
            lst_candidats = set(path_f1rst).intersection(candidates)
            max_conno = 0
            nrn_win_1 = -1
            # print("lst_candidats",lst_candidats)
            for nrn2_id in lst_candidats:
                nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                if max_conno < len(nrn2["DbConnectivity"]["lateral_connexion"]):
                    max_conno = len(nrn2["DbConnectivity"]["lateral_connexion"])
                    nrn_win_1 = nrn2_id
            
            # print("nrn_win 1",nrn_win_1)
            if nrn_win_1!=-1:
                crnt_nrn["meta"]["path"] = [nrn_win_1] + crnt_nrn["meta"]["path"]
                # boucle sur le path du first
                int_pos = 0
                for nrn_id in segmented_path:
                    # regarde les connexions de chaque neurone
                    nrn_connected = self.get_nrn_from_path([nrn_id])
                    if len(nrn_connected.intersection({nrn_win_1}))>0:
                        pos_2_cut_off["nrn_id"].append(nrn_id)
                        pos_2_cut_off["connector"].append(nrn_win_1)
                        pos_2_cut_off["position"].append(int_pos)
                        pos_2_cut_off["crnt_nrn"].append(crnt_nrn)
                        # bl_got_segmented = True
                        print("found possible connexion on nrn id",nrn_id, ":",nrn_connected)
                        break
                    int_pos += 1
            
            tip_2 = path_crnt[len(path_crnt)-1]
            candidates = self.get_nrn_from_path([tip_2])
            lst_candidats = set(path_f1rst).intersection(candidates)
            # print("lst_candidats",lst_candidats)
            max_conno = 0
            nrn_win_2 = -1
            
            for nrn2_id in lst_candidats:
                nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                if max_conno < len(nrn2["DbConnectivity"]["lateral_connexion"]):
                    max_conno = len(nrn2["DbConnectivity"]["lateral_connexion"])
                    nrn_win_2 = nrn2_id

            if nrn_win_2!=-1 and nrn_win_2!=nrn_win_1:
                crnt_nrn["meta"]["path"].append(nrn_win_2)
                int_pos = 0
                for nrn_id in segmented_path:
                    # regarde les connexions de chaque neurone
                    nrn_connected = self.get_nrn_from_path([nrn_id])
                    if len(nrn_connected.intersection({nrn_win_2}))>0:
                        pos_2_cut_off["nrn_id"].append(nrn_id)
                        pos_2_cut_off["connector"].append(nrn_win_2)
                        pos_2_cut_off["position"].append(int_pos)
                        pos_2_cut_off["crnt_nrn"].append(crnt_nrn)
                        # bl_got_segmented = True
                        print("found possible connexion on nrn id",nrn_id, ":",nrn_connected)
                        break
                    int_pos += 1
            
        # Découper le segment principal en commençant par la position la plus petite
        print("Recherche des nœuds suivants", pos_2_cut_off["connector"], pos_2_cut_off["nrn_id"])
        # self.nrn_tls.remove_nrn_by_id(common_nrn)
        print("Commence la découpe de segmented_path",segmented_path)
        while len(pos_2_cut_off["position"])>0:
            int_pos = np.argmin(pos_2_cut_off["position"])
            shorter_path = []
            # print("recherche:",pos_2_cut_off["nrn_id"][int_pos])
            len_path = np.where(np.array(segmented_path) == pos_2_cut_off["nrn_id"][int_pos])
            if np.shape(len_path[0])[0]>0 :
                len_path = len_path[0][0]
            else:
                pos_2_cut_off["nrn_id"].pop(int_pos)
                pos_2_cut_off["connector"].pop(int_pos)
                pos_2_cut_off["position"].pop(int_pos)
                pos_2_cut_off["crnt_nrn"].pop(int_pos)
                continue
            for path_pos in range(len_path+1):
                new_path_nrn_id = segmented_path.pop(0)
                shorter_path.append(new_path_nrn_id)
                if new_path_nrn_id==pos_2_cut_off["nrn_id"][int_pos]:
                    shorter_path.append(pos_2_cut_off["connector"][int_pos])
                    segmented_path = [pos_2_cut_off["connector"][int_pos]] + segmented_path
                    # print("recherche",pos_2_cut_off["nrn_id"][int_pos],"segmented_path",segmented_path)
                    # Create le nrn3
                    nb = self.nrn_tls.add_new_nrn("sentive_sequence_nrn")
                    nrn3 = self.nrn_tls.lst_nrns[nb].neuron
                    nrn3["meta"]["path"] = shorter_path
                    nrn_activated = self.get_nrn_from_path(shorter_path)
                    nrn3["DbConnectivity"]["pre_synaptique"] = nrn_activated
                    nrn3["meta"]["ratio_pxls_total"] = len(nrn_activated)/self.nb_nrn_pxls
                    print("ratio_pxls_total nrn _ID",nrn3["_id"],", ",nrn3["meta"]["ratio_pxls_total"])
                    self.nrn_segments.append(nrn3)
                    bl_got_segmented = True
                    break
            pos_2_cut_off["nrn_id"].pop(int_pos)
            pos_2_cut_off["connector"].pop(int_pos)
            pos_2_cut_off["position"].pop(int_pos)
            pos_2_cut_off["crnt_nrn"].pop(int_pos)

        if bl_got_segmented:
            print("Fin de la découpe de segmented_path",segmented_path)
            self.nrn_segments.pop(0)
            self.nrn_tls.remove_nrn_pos(nb_min)
            nb = self.nrn_tls.add_new_nrn("sentive_sequence_nrn")
            nrn3 = self.nrn_tls.lst_nrns[nb].neuron
            nrn3["meta"]["path"] = segmented_path
            nrn_activated = self.get_nrn_from_path(segmented_path)
            nrn3["DbConnectivity"]["pre_synaptique"] = nrn_activated
            nrn3["meta"]["ratio_pxls_total"] = len(nrn_activated)/self.nb_nrn_pxls
            print("ratio_pxls_total nrn _ID",nrn3["_id"],", ",nrn3["meta"]["ratio_pxls_total"])
            self.nrn_segments.append(nrn3)

        print("nombre de neurones couches 1, 2 et 3 :",len(self.nrn_tls.lst_nrns))
        print("*"*40)


    def layer_3_bis(self):
        for tmp_pos in range(len(self.nrn_segments)):
            if len(self.nrn_segments[tmp_pos]["meta"]["path"])>1:
                # calcul des vecteurs
                nrn2_id = self.nrn_segments[tmp_pos]["meta"]["path"][0]
                nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                # self.nrn_segments[tmp_pos]["ratio_pxls_total"] = len(set(nrn3["meta"]["mobilise_pxl_ids"]))/self.nb_nrn_pxls
                try:
                    point1 = nrn2["meta"]["center"]
                except:
                    continue
                for nrn2_pos in range(1, len(self.nrn_segments[tmp_pos]["meta"]["path"])):
                    nrn2_id = self.nrn_segments[tmp_pos]["meta"]["path"][nrn2_pos]
                    nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                    if nrn2 == '': continue
                    point2 = nrn2["meta"]["center"]
                    vecteur = {
                        "x":point2["x"]-point1["x"],
                        "y":point2["y"]-point1["y"]
                    }
                    self.nrn_segments[tmp_pos]["meta"]["vecteurs"].append(vecteur)
                    self.nrn_segments[tmp_pos]["meta"]["distances"].append(
                        self.nrn_tls.calc_dist(point1, point2))
                    
                    point1 = point2
                
                # calcul des angles
                if len(self.nrn_segments[tmp_pos]["meta"]["vecteurs"])>1:
                    vector_1 = self.nrn_segments[tmp_pos]["meta"]["vecteurs"][0]
                    for v_pos in range(1, len(self.nrn_segments[tmp_pos]["meta"]["vecteurs"])):
                        vector_2 = self.nrn_segments[tmp_pos]["meta"]["vecteurs"][v_pos]
                        self.nrn_segments[tmp_pos]["meta"]["angles"].append(self.nrn_tls.calc_angle(vector_1, vector_2))
                        vector_1 = vector_2



    def layer_4(self):
        """
            Création d'un ensemble de neurones permettant de comparer ensuite les caractères
        """
        # création d'une nouvelle couche 
        self.nrn_tls.new_layer()

        len_nrn = len(self.nrn_tls.lst_nrns)
        for nrn3_pos in range(self.nrn_tls.pos_nrn_by_layer[2], len_nrn):
            nrn3 = self.nrn_tls.lst_nrns[nrn3_pos].neuron
            # check 
            if nrn3["layer_id"]==3:
                for nrn2_pos in range(1, len(nrn3["meta"]["path"])-1):
                    nrn2_id = nrn3["meta"]["path"][nrn2_pos]
                    # crée un nouveau neurone
                    self.new_angle_neuron(nrn3["meta"]["path"][nrn2_pos-1], nrn2_id, nrn3["meta"]["path"][nrn2_pos+1])

                for nrn3_pos2 in range(nrn3_pos+1, len_nrn):
                    nrn3_2 = self.nrn_tls.lst_nrns[nrn3_pos2].neuron
                    # print(nrn3_2)
                    # Vérifie maintenant les 4 extrémités
                    if nrn3["meta"]["path"][0] == nrn3_2["meta"]["path"][0]:
                        # Crée un neurone commun
                        self.new_angle_neuron(nrn3["meta"]["path"][1], nrn3["meta"]["path"][0], nrn3_2["meta"]["path"][1])
                    elif nrn3["meta"]["path"][0] == nrn3_2["meta"]["path"][-1]:
                        self.new_angle_neuron(nrn3["meta"]["path"][1], nrn3["meta"]["path"][0], nrn3_2["meta"]["path"][-2])

                    elif nrn3["meta"]["path"][-1] == nrn3_2["meta"]["path"][-1]:
                        self.new_angle_neuron(nrn3["meta"]["path"][-2], nrn3["meta"]["path"][-1], nrn3_2["meta"]["path"][-2])
                    elif nrn3["meta"]["path"][-1] == nrn3_2["meta"]["path"][0]:
                        self.new_angle_neuron(nrn3["meta"]["path"][-2], nrn3["meta"]["path"][-1], nrn3_2["meta"]["path"][1])
        
        # Création des connexions latérales
        len_nrn = len(self.nrn_tls.lst_nrns)
        for nrn4_pos in range(self.nrn_tls.pos_nrn_by_layer[3], len_nrn):
            nrn4 = self.nrn_tls.lst_nrns[nrn4_pos].neuron
            # check 
            if nrn4["layer_id"]==4:
                for nrn2_id in nrn4["DbConnectivity"]["pre_synaptique"]:
                    nrn2 = self.nrn_tls.get_neuron_from_id(nrn2_id)
                    for nrn_lateral_id in nrn2["DbConnectivity"]["post_synaptique"]:
                        self.nrn_tls.add_nrn_lateral(nrn4, nrn_lateral_id)


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
        ##self.layer_2_v2() # triplets
        # self.layer_2() # triplets
        # self.couche_3()
        # self.show_layer_vectors(3, False)
        # self.calc_angles_layer_3()
        # self.show_vectors_directions(3, False)
        # self.plot_angles_3()
        # self.layer_3() # séquences, segments
        # self.layer_3_bis() # calcul des angles
        # self.layer_4() # binomes -> caractères


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