import numpy as np
import copy

from .sentive_vision_network import sentive_vision_network
 
class sentive_brain():
    def __init__(self, episode, nb_char):
        self.episode = episode
        self.nnet = []
        self.nb_char = nb_char
        for i in range(nb_char):
            print("\n$ >********* network:",i)
            self.nnet.append(sentive_vision_network(episode[0][i]))
            self.nnet[i].run_layers()


    def predict(self, test_img):
        self.test_net = sentive_vision_network(test_img)
        self.test_net.run_layers()
        full_results = []
        for lnet in range(self.nb_char):
            # pour chaque caractère je fais une boucle sur chaque binome
            full_results.append(self.nnet[lnet].activate_with_input(self.test_net))

            # return full_results
            lst_local_curve_nrns, local_curve_total_length = self.nnet[lnet].get_curve_nrn_list()
            lst_input_curve_nrns, input_curve_total_length = self.test_net.get_curve_nrn_list()
            tmp_results = []
            tmp_line_results = []
            # calculer le centre de gravité de la courbe à partir de meta.pixels_matrix
            for input_curve_nrn in lst_input_curve_nrns:
                nrn = input_curve_nrn.neuron
                if len(nrn["meta"]["curve"]["pixels_matrix"])<1:
                    nrn["meta"]["curve"]["pixels_matrix"] = self.test_net.nrn_tls.get_list_pixels_coord(nrn["_id"])
                    # print("input nrn", nrn["_id"], "pixels_matrix:", nrn["meta"]["curve"]["pixels_matrix"])
                # calculer le centre de gravité de la courbe à partir de meta.pixels_matrix
                list_x = []
                list_y = []
                for coord in nrn["meta"]["curve"]["pixels_matrix"]:
                    list_x.append(coord["x"])
                    list_y.append(coord["y"])
                nrn["meta"]["curve"]["cg"] = {
                    "x":np.mean(list_x),
                    "y":np.mean(list_y)
                }
                nrn["meta"]["curve"]["offset_cg"] = {
                    "x":nrn["meta"]["curve"]["cg"]["x"]-self.test_net.glbl_prm["cg"]["x"],
                    "y":nrn["meta"]["curve"]["cg"]["y"]-self.test_net.glbl_prm["cg"]["y"]
                }

            for local_curve_nrn in lst_local_curve_nrns:
                nrn = local_curve_nrn.neuron
                if len(nrn["meta"]["curve"]["pixels_matrix"])<1:
                    nrn["meta"]["curve"]["pixels_matrix"] = self.nnet[lnet].nrn_tls.get_list_pixels_coord(nrn["_id"])
                    # print("local nrn", nrn["_id"], "pixels_matrix:", nrn["meta"]["curve"]["pixels_matrix"])
                # calculer le centre de gravité de la courbe à partir de meta.pixels_matrix
                list_x = []
                list_y = []
                for coord in nrn["meta"]["curve"]["pixels_matrix"]:
                    list_x.append(coord["x"])
                    list_y.append(coord["y"])
                nrn["meta"]["curve"]["cg"] = {
                    "x":np.mean(list_x),
                    "y":np.mean(list_y)
                }
                nrn["meta"]["curve"]["offset_cg"] = {
                    "x":nrn["meta"]["curve"]["cg"]["x"]-self.nnet[lnet].glbl_prm["cg"]["x"],
                    "y":nrn["meta"]["curve"]["cg"]["y"]-self.nnet[lnet].glbl_prm["cg"]["y"]
                }

            # calculer les scores
            lst_local_sequence = self.nnet[lnet].get_lateral_nb_previous_nrns(lst_local_curve_nrns)
            activated_sequence = np.zeros(len(lst_local_sequence))
            final_results = np.zeros(len(lst_local_sequence))
            for input_curve_nrn in lst_input_curve_nrns:
                for tmp_counter, local_curve_nrn in enumerate(lst_local_curve_nrns):
                    if lst_local_sequence[tmp_counter]>0:
                        tmp_propagated_value = activated_sequence[tmp_counter]/lst_local_sequence[tmp_counter]
                    else:
                        tmp_propagated_value = 0
                    # print(tmp_propagated_value)
                    tmp_results.append(self.nnet[lnet].get_single_curve_activation(input_curve_nrn, local_curve_nrn, tmp_propagated_value, local_curve_total_length, input_curve_total_length))
                    best_line_result = 0
                    for local_line_id in local_curve_nrn.neuron["DbConnectivity"]["anti_post_synaptique"]:
                        for input_line_id in input_curve_nrn.neuron["DbConnectivity"]["anti_post_synaptique"]:
                            input_line_nrn = self.test_net.nrn_tls.get_neuron_from_id(input_line_id)
                            # tmp_line_results.append(get_single_line_activation(self, local_curve_nrn, input_curve_nrn.neuron["DbConnectivity"]["anti_post_synaptique"][0]))
                            tmp_line_result = self.nnet[lnet].get_single_line_activation(self, input_line_nrn, local_line_id)
                            if best_line_result < tmp_line_result:
                                best_line_result = copy.deepcopy(tmp_line_result)
                    tmp_line_results.append(best_line_result)
                    
                    # the winner is the one with the highest score
                    # the winner takes it all
                # print("tmp_line_results", tmp_line_results)
                # print("tmp_results", tmp_results)
                for id in range(len(tmp_results)):
                    tmp_results[id] = tmp_results[id] * tmp_line_results[id]

                id_winner = np.argmax(tmp_results)
                # print(tmp_results)
                # print("winner:", id_winner, "score:", tmp_results[id_winner], "nrn3", lst_local_curve_nrns[id_winner].neuron["_id"])
                tmp_result_id = tmp_results[id_winner] * 0.5*((input_curve_nrn.neuron["meta"]["curve"]["nb_iteration"]/input_curve_total_length)+(lst_local_curve_nrns[id_winner].neuron["meta"]["curve"]["nb_iteration"]/local_curve_total_length))
                if final_results[id_winner] < tmp_result_id:
                    final_results[id_winner] = tmp_result_id


                # propagate the activity
                bool_propagated_activity = False
                # print("Before activated_sequence", activated_sequence)
                new_max_propagated_value = 0
                for id in range(len(activated_sequence)):
                    if id==id_winner:
                        new_max_propagated_value = activated_sequence[id] + 1
                        activated_sequence[id] = 0
                        # check if winner has a lateral connexion
                        if len(lst_local_curve_nrns[id_winner].neuron["DbConnectivity"]["lateral_connexion"])>0:
                            bool_propagated_activity = True
                    elif bool_propagated_activity:
                        activated_sequence[id]+= 1
                        if activated_sequence[id] > new_max_propagated_value:
                            activated_sequence[id] = new_max_propagated_value
                        if len(lst_local_curve_nrns[id].neuron["DbConnectivity"]["lateral_connexion"])>0:
                            bool_propagated_activity = True
                        else:
                            bool_propagated_activity = False
                    else:
                        activated_sequence[id] = 0
                    
                # print("After activated_sequence", activated_sequence)
                tmp_results = []
                tmp_line_results = []

            # print("final_results", final_results)
            # print("result", np.sum(final_results))
            full_results.append(np.sum(final_results))
        return full_results
