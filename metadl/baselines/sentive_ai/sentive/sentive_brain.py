import numpy as np
import pandas as pd

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
            # self.nnet[lnet]
            full_results.append(self.nnet[lnet].activate_with_input(self.test_net))

        return full_results

