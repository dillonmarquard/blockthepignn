
# coding: utf-8

# In[ ]:


from PIL import Image
import time
import numpy as np
from keras.models import load_model
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import tensorflow as tf
import random
import threading
from IPython.display import clear_output
from keras.utils import to_categorical
import pygame
from pygame.locals import *
import math
import time


# In[ ]:


class Game():
    def __init__(self):
        self.pig = np.zeros(55)
        self.pig[27] = 1
        self.stone = np.zeros(55)
        self.grass = np.zeros(55)
        self.gameBoard = None

    def generateGameBoard(self, difficulty):  # 1(easy) to 5(hard)
        if difficulty > 4: #max difficulty so nn can train easier
            difficulty = 4
        self.randomGameBoard(difficulty)
        self.separateGameBoard()

    def randomGameBoard(self,difficulty):
        self.gameBoard = np.zeros(55)
        for x in range(55):
            if random.random() > (32/55 + (1/55 * difficulty)):
                    self.gameBoard[x] = 1
        self.gameBoard[27] = 2
        nextMov, pigLoc = self.pigNextMoveFF()
        if nextMov == None:
            self.randomGameBoard(difficulty)

    def separateGameBoard(self):
        # copies the current gameBoard onto the individual pig and stone arrays
        self.stone = np.zeros(55)
        self.pig = np.zeros(55)
        self.grass = np.zeros(55)
        for x in range(55):
            if self.gameBoard[x] == 1:
                self.stone[x] = 1
            if self.gameBoard[x] == 2:
                self.pig[x] = 1
            if self.gameBoard[x] == 0:
                self.grass[x] = 1
        self.input = []
        self.input.extend(self.stone)
        self.input.extend(self.pig)
        self.input.extend(self.grass)
        self.input = np.array([self.input])
        
    def printArr(self, arr):
        for x in range(11):
            if ((x + 1) % 2) == 0:
                y = "  "
            else:
                y = ""
            print(y, arr[(x*5):(x*5)+5])

    def pigNextMoveFF(self):
        newLoc = None
        fillPath = np.zeros(55)
        for x in range(55):
            if self.gameBoard[x] == 2:
                pigLoc = x
                break

        # if self.pigLoc < 5 or self.pigLoc > 49 or self.pigLoc % 5 == 0 or (self.pigLoc - 4) % 5 == 0: ## this includes all the edges of the gameBoard array
            # return 69 # returns none if pig is on edge

        fillPath[pigLoc] = 1
        for iteration in range(12):  # flood fill
            for loc in range(55):
                if fillPath[loc] == iteration + 1:
                    if int(str(loc)[len(str(loc)) - 1]) < 6:
                        x = -1
                    else:
                        x = 0
                    if (loc + 1) < 55 and (loc + 1) >= 0 and self.gameBoard[loc + 1] == 0 and fillPath[loc + 1] == 0:
                        fillPath[loc + 1] = iteration + 2
                    if (loc - 1) < 55 and (loc - 1) >= 0 and self.gameBoard[loc - 1] == 0 and fillPath[loc - 1] == 0:
                        fillPath[loc - 1] = iteration + 2
                    if (loc + 5 + x) < 55 and (loc + 5 + x) >= 0 and self.gameBoard[loc + 5 + x] == 0 and fillPath[loc + 5 + x] == 0:
                        fillPath[loc + 5 + x] = iteration + 2
                    if (loc + 6 + x) < 55 and (loc + 6 + x) >= 0 and self.gameBoard[loc + 6 + x] == 0 and fillPath[loc + 6 + x] == 0:
                        fillPath[loc + 6 + x] = iteration + 2
                    if (loc - 4 + x) < 55 and (loc - 4 + x) >= 0 and self.gameBoard[loc - 4 + x] == 0 and fillPath[loc - 4 + x] == 0:
                        fillPath[loc - 4 + x] = iteration + 2
                    if (loc - 5 + x) < 55 and (loc - 5 + x) >= 0 and self.gameBoard[loc - 5 + x] == 0 and fillPath[loc - 5 + x] == 0:
                        fillPath[loc - 5 + x] = iteration + 2
            for loc in range(55):
                if loc < 5 or loc > 49 or loc % 5 == 0 or (loc - 4) % 5 == 0:
                    if fillPath[loc] > 0:  # next move found

                        loc1 = loc
                        count = 0
                        while fillPath[loc1] != 2:
                            count += 1
                            if count > 200:
                                break
                                #print(
                                #    "value of fillPath[loc1]: ", fillPath[loc1], " value: ", loc1)
                                #game.printArr(fillPath)
                            if int(str(loc1)[len(str(loc1)) - 1]) < 5:
                                x = -1
                            else:
                                x = 0
                            if (loc1 + 1) < 55 and (loc1 + 1) >= 0 and fillPath[loc1 + 1] < fillPath[loc1] and fillPath[loc1 + 1] > 0 and (loc - 4) % 5 != 0:
                                loc1 = loc1 + 1
                                continue
                            if (loc1 - 1) < 55 and (loc1 - 1) >= 0 and fillPath[loc1 - 1] < fillPath[loc1] and fillPath[loc1 - 1] > 0 and loc % 5 != 0:
                                loc1 = loc1 - 1
                                continue
                            if (loc1 + 5 + x) < 55 and (loc1 + 5 + x) >= 0 and fillPath[loc1 + 5 + x] < fillPath[loc1] and fillPath[loc1 + 5 + x] > 0:
                                loc1 = loc1 + 5 + x
                                continue
                            if (loc1 + 6 + x) < 55 and (loc1 + 6 + x) >= 0 and fillPath[loc1 + 6 + x] < fillPath[loc1] and fillPath[loc1 + 6 + x] > 0:
                                loc1 = loc1 + 6 + x
                                continue
                            if (loc1 - 4 + x) < 55 and (loc1 - 4 + x) >= 0 and fillPath[loc1 - 4 + x] < fillPath[loc1] and fillPath[loc1 - 4 + x] > 0:
                                loc1 = loc1 - 4 + x
                                continue
                            if (loc1 - 5 + x) < 55 and (loc1 - 5 + x) >= 0 and fillPath[loc1 - 5 + x] < fillPath[loc1] and fillPath[loc1 - 5 + x] > 0:
                                loc1 = loc1 - 5 + x
                                continue
                        newLoc = loc1
                        # 0 - 54 related to the gameBoard array index # the current location of the pig
                        return newLoc, pigLoc
                # else:
                    # if loc == 55: # last point in array checked so if even the last index has a value of zero then there are no paths and player wins

                        # return newLoc # no next move means player wins # still return pigLoc for continuity of return within the function
        newLoc = None
        return newLoc, pigLoc

    def play(self, model):
        model.fitness = 0
        model.level = 0
        gameswon = 0
        for x in range(100):
            #print("new game")
            model.inputlist = []
            model.outputlist = []
            self.generateGameBoard(model.level)
            self.move(model.getPrediction(self),model)
            self.move(model.getPrediction(self),model)
            self.move(model.getPrediction(self),model)
            
            while True:
                newLoc, pigLoc = self.pigNextMoveFF()
                if newLoc == None:
                    #print("player wins")
                    model.level += 1
                    gameswon += 1
                    #print("len of input list: ",len(model.inputlist))
                    for each in population.Models:
                        for x in range(len(model.inputlist)):
                            each.model.train_on_batch(model.inputlist[x], model.outputlist[x], sample_weight=None, class_weight=None)
                    break
                else:
                    self.gameBoard[newLoc] = 2
                    self.gameBoard[pigLoc] = 0
                    #print("pig moves")
                    if newLoc < 5 or newLoc > 49 or newLoc % 5 == 0 or (newLoc - 4) % 5 == 0:
                        #("pig wins")
                        for x in range(len(model.inputlist)):
                            total = 55
                            for y in range(55):
                                if model.inputlist[x][0][y+110] != 1:
                                    total -= 1
                                    model.outputlist[x][0][y] = 0
                            for y in range(55):
                                if model.inputlist[x][0][y+110] == 1:
                                    if model.outputlist[x][0][y] == 1:
                                        model.outputlist[x][0][y] = 0
                                    else:
                                        model.outputlist[x][0][y] = 1/total
                        for each in population.Models:
                            for x in range(len(model.inputlist)):
                                each.model.train_on_batch(model.inputlist[x], model.outputlist[x], sample_weight=None, class_weight=None)
                        model.fitness += model.level * 5
                        break
                    else:
                        self.separateGameBoard()
                        self.move(model.getPrediction(self),model)
            
        print("games won: ",gameswon,"/100")
        model.fitness = model.fitness/100
        return 

    def move(self, index, model):  # updates all the game classes attributes
        #print("player moves")
        # make sure it doesnt write over the pig which would be bad...
        if(self.gameBoard[index]) == 0:
            self.gameBoard[index] = 1
            model.fitness += .1
            outputarr = np.zeros(55)
            outputarr[index] = 1
            outputarr = np.array([outputarr])
            model.inputlist.append(self.input)
            model.outputlist.append(outputarr)
            #print("cont input len: ",len(model.inputlist))
        else:
            model.fitness += -.1
        self.separateGameBoard()
        # self.printArr(self.gameBoard)


# In[ ]:


class Models:
    def __init__(self, model=None, mutationRate=0):
        if(model == None):
            self.model = Sequential()
            self.model.add(Dense(165, input_shape=(165,), activation="relu", use_bias=False))
            self.model.add(Dense(55, activation="softmax", use_bias=False))
            
            #Dense(25, activation="relu", use_bias=False),
            #Dense(25, activation="relu", use_bias=False),
            
        else:
            self.model = model
        self.model.compile(Adam(lr=.00012), loss='mean_squared_error', metrics=['accuracy'])
        self.mutationRate = mutationRate
        self.level = 0
        self.fitness = None
        self.inputlist = []
        self.outputlist = []
        
    def mutate(self, rand):  # rand is the % chance to mutate
        mutatedModel = self.model  # copy of model
        mutatedWeights = mutatedModel.get_weights()
        # for each model a % of connections on average are pseudo-randomly changed by a random value between 0 and .1
        for layer in range(len(mutatedModel.get_weights())):
            for node in range(len(mutatedModel.get_weights()[layer])):
                for weight in range(len(mutatedModel.get_weights()[layer][node])):
                    if (random.random() < (rand / 100)):
                        if random.random() < .5:
                            mutatedWeights[layer][node][weight] += .0001
                        else:
                            mutatedWeights[layer][node][weight] -= .0001
        mutatedModel.set_weights(mutatedWeights)

        # returns a new model class with the mutated weights
        return Models(model=mutatedModel)

    def getPrediction(self, game):
        arr = []
        arrn = []
        arr.extend(game.stone) # 0-54
        arr.extend(game.pig)   # 55-109
        arr.extend(game.grass) # 110-165
        arrn = np.array([arr])
        predictions = self.model.predict(arrn, batch_size=1)
        #print(predictions)
        #debug make stones zero probability, test 
        #for x in range(55):
        #    if game.gameBoard[x] == 1 or game.gameBoard[x] == 2:
        #        predictions[0, x] = 0
        for x in range(55):
            if predictions[0, x] == predictions.max():
                maxLocation = x
                #print(maxLocation)
                return maxLocation

    def crossover(self, other):
        mom = self.model  # copy of model
        momweights = mom.get_weights()
        dad = other.model  # copy of other model
        dadweights = dad.get_weights()
        for layer in range(len(mom.get_weights())):
            for node in range(len(mom.get_weights()[layer])):
                for weight in range(len(mom.get_weights()[layer][node])):
                    if random.random() < .5:
                        momweights[layer][node][weight] = dadweights[layer][node][weight]
        mom.set_weights(momweights)

        # returns a new model class with the baby weights and slight mutations
        return Models(model=mom)#.mutate(self.mutationRate)


# In[ ]:


class Population:
    def __init__(self):
        self.Models = []
        self.Threads = []
        
    def Populate(self, pop_size):  # possible threading
        initialgentime = time.time()
        for x in range(pop_size):
            self.Models.append(Models())
        print("repopulating")
        for each in self.Models:
            if each.fitness == None:
                game.play(each)
        print("creation time: ","%.3f" %  (time.time() - initialgentime))
        print("")
    def bubbleSort(self):
        for j in range((len(self.Models) - 1)):
            for i in range((len(self.Models) - j - 1)):
                if self.Models[i].level < self.Models[i+1].level:
                    temp = self.Models[i]
                    self.Models[i] = self.Models[i+1]
                    self.Models[i+1] = temp

    def saveModels(self,numToSave=100): # from 0 to (numToSave - 1)                
        for x in range(numToSave):
            self.Models[x].model.save("models/model" + str(x) + ".h5")
        print("finished saving models")
    
    def loadModels(self,numToLoad): # appends the model to the Models array
        for x in range(numToLoad):
            print(x)
            self.Models.append(Models(model=load_model("models/model" + str(x) + ".h5")))           
        for each in self.Models:
            game.play(each)
        print("finished loading models")
    
    def runGeneticAlgo(self, epochs, game=Game(), population=0):
        if population != 0:
            self.Populate(population)
        modelsSize = len(self.Models)
        
        for epoch in range(epochs):
            epochtime = time.time()
            #for x in range(modelsSize - 1):
            #    self.Models.append(self.Models[x].crossover(self.Models[random.randint(0, modelsSize - 1)]))
            #for x in range(modelsSize//5):
            #    self.Models[x] = self.Models[x].mutate(25)
            
            for each in self.Models: # possible threading
                #if each.fitness == None:
                game.play(each)
            self.bubbleSort()
            
            #print("purging weak")
            while len(self.Models) > modelsSize:
                self.Models.pop()
            
            sumFitness = 0
            for x in range(len(self.Models)):
                sumFitness += self.Models[x].fitness
            avgFitness = sumFitness / len(self.Models)
            
            #numOfBabies = 0
            #for x in range(len(self.Models)): # not scalable with limited resources
            #    if self.Models[len(self.Models) - x - 1].fitness < avgFitness and numOfBabies < modelsSize // 5:# or self.Models[x].fitness < 0:
            #        self.Models[len(self.Models) - x - 1] = Models()
            #        numOfBabies += 1
            #for each in self.Models: # possible threading
            #    if each.fitness == None:
            #        game.play(each)
            #self.bubbleSort()
            if epoch % 25 == 0 and epoch != 0:
                trainAgainstStone()
            if epoch % 100 == 0 and epoch != 0:
                clear_output()
                
            print("epoch: ", epoch)
            print("epoch time: ", "%.3f" % (time.time() - epochtime))
            print("best level: ",self.Models[0].level," worst level: ",self.Models[len(self.Models) - 1].level,"best fitness: ",'%.3f' % self.Models[0].fitness," worst fitness: ",'%.3f' % self.Models[len(self.Models) - 1].fitness," avg fitness: ","%.3f" % avgFitness)
            print("population size: ", len(self.Models)," randomized: ")#numOfBabies)
            print("")
            
            if epoch % 50 == 0 and epoch != 0:
                self.saveModels(numToSave=modelsSize)
        self.saveModels(numToSave=modelsSize)


# In[ ]:


def outPutArrayGen(gameBoard):
    total = 0
    outarr = np.zeros(55)
    for x in range(55):
        if gameBoard[x] == 0:
            total += 1
    for x in range(55):
        if gameBoard[x] == 0:
            outarr[x] = 1/total
        else:
            outarr[x] = 0
    return outarr


# In[ ]:


#train to not click on stone
def trainAgainstStone():
    for x in range(200):
        for each in population.Models:
            game.generateGameBoard(random.randint(0,15))
            outputarr = outPutArrayGen(game.gameBoard)
            outputarr = np.array([outputarr])
            each.model.train_on_batch(game.input, outputarr, sample_weight=None, class_weight=None)
        clear_output()
        print(x)
    population.saveModels(numToSave=len(population.Models))


# In[ ]:


with tf.device('/gpu:0'):
    game = Game()
    population = Population()
    population.loadModels(numToLoad=15)
    #population.Populate(50)
    #trainAgainstStone()
    population.bubbleSort()


# In[ ]:


with tf.device('/gpu:0'):
    population.runGeneticAlgo(700,population=0) # currently only plays the game for reinforcement learning


# In[ ]:


#bestlevel = 0
#for x in range(100):
#    print("epoch: ",(x+1),"/100")
#    for y in range(400):
#        for each in population.Models:
#            game.play(each)
#            print("model fitness: ",'%.3f' % each.fitness)
#            print("")
#    #population.bubbleSort()
#        if y % 150 == 0:
#            trainAgainstStone()
#            population.saveModels(numToSave=len(population.Models))
#        for each in population.Models:
#            if each.level > bestlevel:
#                bestlevel = each.level
#        print("")
#        print("best model: ",bestlevel,"/100")
#            #print("model level: ",each.level)
#        print("")
#        
#    clear_output()
#population.saveModels(numToSave=len(population.Models))
#clear_output()
#print("end of session")


# In[ ]:


#game.generateGameBoard(0)
#print(population.Models[0].getPrediction(game))
#game.printArr(game.gameBoard)
#arr = []
#arrn = []
#arr.extend(game.stone)
#arr.extend(game.pig)
#arr.extend(game.grass)
#arrn = np.array([arr])
#predictions = population.Models[0].model.predict(arrn, batch_size=1)
#print(predictions)


# In[ ]:


pygame.init()
screen_width=300
screen_height=580
screen=pygame.display.set_mode([screen_width,screen_height])
pygame.display.init()
def SetCirclePostions():
    Positions = []
    for y in range(11):
        for x in range(5):
            if y%2 == 0:
                Positions.append([x*50 + 30, y*50 + 30])
            else:
                Positions.append([x*50 + 55, y*50 + 30])
    return Positions
def DrawGame(GameBoard):
    for y in range(11):
        for x in range(5):
            if GameBoard[x+y*5] == 1:
                Color = (100,100,100)
            elif GameBoard[x+y*5] == 2:
                Color = (100,0,0)
            else:
                Color = (0,100,0)
            if y%2 == 0:
                pygame.draw.circle(screen,Color,(x*50 + 30, y*50+30),25)
            else:
                pygame.draw.circle(screen,Color,(x*50 + 55, y*50+30),25)
    #pygame.display.flip()
def GetSpaceClicked(Pos):
    Current = 0
    for Location in Pos:
        points = (pygame.mouse.get_pos(),Location)
        p0, p1 = points
        if abs(math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)) <= 20:
            #print(Current)
            return Current
        else:
            Current += 1
    return None
def PlayGame():
    
    Pos = SetCirclePostions()
    #print(Pos)
    MousePressed = False
    DrawGame(game.gameBoard)
    clock = pygame.time.Clock()
    lastState = 1
    running = True
    turns = 3
    while True:
        population.Models[0].inputlist = []
        population.Models[0].outputlist = []
        game.generateGameBoard(population.Models[0].level)
        game.separateGameBoard()
        botpredictionloc = population.Models[0].getPrediction(game)
        print("newgame")
        while running == True:
            
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    running = False
                    break
            if pygame.mouse.get_pressed()[0] == 1 and lastState == 0:
                lastState = pygame.mouse.get_pressed()[0]
                Location = GetSpaceClicked(Pos)
                game.separateGameBoard()
                clear_output()
                print(turns)
                if Location != None and game.gameBoard[Location] == 0:
                    if game.gameBoard[Location] == 0:
                        outputarr = np.zeros(55)
                        outputarr[int(Location)] = 1
                        outputarr = np.array([outputarr])
                        population.Models[0].inputlist.append(game.input)
                        population.Models[0].outputlist.append(outputarr)
                        for x in range(10):
                            population.Models[0].model.train_on_batch(game.input, outputarr, sample_weight=None, class_weight=None)
                        game.gameBoard[Location] = 1
                        turns -= 1
                        if turns <= 0:
                            newLoc, pigLoc = game.pigNextMoveFF()
                            if newLoc == None:
                                print("player wins")
                                #print(len(population.Models[0].inputlist))
                                for x in range(len(population.Models[0].inputlist)):
                                    population.Models[0].model.train_on_batch(population.Models[0].inputlist[x], population.Models[0].outputlist[x], sample_weight=None, class_weight=None)
                                    #print(population.Models[0].inputlist[x],"",population.Models[0].outputlist[x])
                                if population.Models[0].level < 5:
                                    population.Models[0].level += 1
                                turns = 3
                                #game.generateGameBoard(population.Models[0].level)
                                break
                            else:
                                game.gameBoard[newLoc] = 2
                                game.gameBoard[pigLoc] = 0
                                #print("pig moves")
                                if newLoc < 5 or newLoc > 49 or newLoc % 5 == 0 or (newLoc - 4) % 5 == 0:
                                    print("pig wins")
                                    ##
                                    for x in range(len(model.inputlist)):
                                        total = 55
                                        for y in range(55):
                                            if model.inputlist[x][0][y+110] != 1:
                                                total -= 1
                                                model.outputlist[x][0][y] = 0
                                    for y in range(55):
                                        if model.inputlist[x][0][y+110] == 1:
                                            if model.outputlist[x][0][y] == 1:
                                                model.outputlist[x][0][y] = 0
                                            else:
                                                model.outputlist[x][0][y] = 1/total
                                    for each in population.Models:
                                        for x in range(len(model.inputlist)):
                                            each.model.train_on_batch(model.inputlist[x], model.outputlist[x], sample_weight=None, class_weight=None)
                                    ##
                                    if population.Models[0].level > 0:
                                        population.Models[0].level -= 1
                                    turns = 3
                                    #game.generateGameBoard(population.Models[0].level)
                                    break
                    game.separateGameBoard() 
                    botpredictionloc = population.Models[0].getPrediction(game)
            DrawGame(game.gameBoard)
            if (botpredictionloc%11)%2 == 0:
                pygame.draw.circle(screen,(0,0,100),((botpredictionloc%5)*50 + 30, (botpredictionloc%11)*50+30),10)
            else:
                pygame.draw.circle(screen,(0,0,100),((botpredictionloc%5)*50 + 55, (botpredictionloc%11)*50+30),10)
            pygame.display.flip()
            lastState = pygame.mouse.get_pressed()[0]
PlayGame()

