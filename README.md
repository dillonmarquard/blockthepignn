## blockthepignn - Neural Network to beat Block the Pig on Cool Math Games
### Prerequisites
* Keras(up-to-date)
* imports may require pip install
* Jupyter notebook is recommended if you plan on contributing to the project
### Setup
* It is recommended to play the game on Cool Math Games to better understand the goal and mechanics of the game.
* Yes we know that a tree search could be used but this is an interesting appplication and great place for many aspiring coders to test different networks and theories
* It is recommended when beginning training to pretrain all your models using the method "trainAgainstStone()" to avoid the initial 
* Feel free to use the **pretrained models**, which as of 5/16/2019 have a **50-70% win-rate**
* If using the pretrained models the folder that it pulls models from is currently /models/... however if you want to use the archived models simply change the name of the folder to "models" since it iterates from 0 to x(number of models to load)
### Other information
* As of 5/16/2019 there seems to be a local minimum that is preventing the nn from passing the 70% threshold and getting higher win-rates this may be from the simplicity of the network or lack of intuative design, but help is appreciated
