# -*- coding: utf-8 -*-

# AI for self driving car

# importing libraries

import numpy as nm
import random
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#class pour réseaux de neuronnes
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
    #fonction pour la propagation
    def forward(self, state):
        #x est la couche cachée
        #F.relu est la fonction d'activation
        x = F.relu(self.fc1(state)) 
        # on calcule les prédictions (les q)
        q_values = self.fc2(x)
        return q_values

#expererience replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        #event = transition
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            #alors on suprrime la plus ancienne transition(event)
            del self.memory[0]
    def sample(self, batch_size):
        #batch_size = nbre de transitions à piocher
        #avec zip on groupe les transitions par catégorie 
        samples = zip(*random.sample(self.memory, batch_size))
        
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
        
    
# implementing Deep Q-Learning
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        #gamma = facteur de réduction
        self.model = Network(input_size, nb_action)
        self.gamma = gamma
        self.reward_window = [] #c'est la moyenne des 100 dernières récompenses
        #reward_window contient les 100 dernières récompenses
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        #lr = learning rate qui correspond au taux d'apprentissage
        #qui lui meme correspond au alpha de la Diff Temporelle
        #last state = l'etat actuel de l'AI
        self.last_state = torch.Tensor(input_size).unsqueeze(0); #ajouter une dimension
        self.last_action = 0
        self.last_reward = 0 

    def select_action(self, state):
        probs = F.softmax(self.model(state) * 0 , dim=1 )
        action = probs.multinomial(num_samples=1) #choisir la meilleure probabilité
        return action.data[0, 0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        #ci dessous l'équation de belleman
        targets = self.gamma*next_outputs + batch_reward
        #targets sont les cibles
        #ci dessous on calcule la fonction de cout
        #td_less ici représente la difference temporelle
        td_loss = F.smooth_l1_loss(outputs, targets) #nous calcule les couts 
        #en dessous on fait de la retropropagation
        #la première etape consite à initialiser notre optimizer
        self.optimizer.zero_grad() #vide le gradient qui a été calculé les fois precedentes
        #la fonction de cout possède une methode backward
        td_loss.backward() #pour faire la retropropagation
        #on enregistre les résultats dans l'optimizer pour mettre à jour les poids
        self.optimizer.step()
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            #tout de suite en bas on doit appeler la methode learn pour mettre à jour les poids
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        
        self.last_action = action
        self.last_state = new_state
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            
        return action
    
    def score(self):
        #cette fonction va nous aider à savoir si l'intélligence apprends ou pas
        #on retourne la moyenne
        return sum(self.reward_window)/(len(self.reward_window) + 1.)
    
    def save(self):
        #state_dict() recupère tous les poids à l'intérieur du modèle
        #save prends un dictionnaire en paramètre
        torch.save({
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
                }, "last_bran.pth")
    
    def load(self):
        if os.path.isfile("last_brain.pth"):
            print("=> loading checkpoing ...")
            checkpoint = torch.load("last_brain.pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("done !")
        else:
            print("no checkpoint found...")
            
        
        







































        
