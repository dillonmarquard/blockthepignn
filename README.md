## blockthepignn - Neural Network to beat Block the Pig on Cool Math Games
### Prerequisites
* Keras(up-to-date) rollback version if softmax is requiring an axis that is a bug and makes the version unusuable.
* imports may require pip install
* Jupyter notebook is recommended if you plan on contributing to the project
### Setup
* It is recommended to play the game on Cool Math Games to better understand the goal and mechanics of the game.
* Yes we know that a tree search could be used but this is an interesting appplication and great place for many aspiring coders to test different networks and theories
* It is recommended when beginning training to pretrain all your models using the method "trainAgainstStone()" to avoid the initial 
* Feel free to use the **pretrained models**, which as of 5/16/2019 have a **80-96% win-rate**
* If using the pretrained models the folder that it pulls models from is currently /models/... however if you want to use the archived models simply change the name of the folder to "models" since it iterates from 0 to x(number of models to load)
### Other information
* As of 5/16/2019 there seems to be a local minimum that is preventing the nn from passing the 70% threshold and getting higher win-rates this may be from the simplicity of the network or lack of intuative design, but help is appreciated
* As of 6/16/2019 the implementation of an exploration based model, which tests each model on one-hundred randomly generated gameboards five times each gameboard with the first two tests and the fourth test having a 1/12 chance of randomly guessing an index in the gameboard(from 0 to 54 pseudorandom integer) and the third and fifth test being the direct max value in the softmax prediction, has given the model the capability to grow beyond its previous record of 92% win-rate up to 96%.
* As of 6/16/2019 It is unknown whether all gameboards are feasible, such that since the gameboards are pseudorandomly generated with a statistically decreasing number of stones that it is feasible that too few stones are placed and no solutions are possible. While the implementation of recording the gameboards that are least completed in third and fifth tests is possible to determine their feasibility since this would just be a statistic and would not benefit the progress of the model it is as of this point in the development irrelevant.
