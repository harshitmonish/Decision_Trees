# Project Title:

## To predicting whether a person is rich (1) or not (0) based on various personal attributes using the decision tree algorithm

### In this problem, we will work with the [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult) available on the UCI repository. The first row in each file specifies the type of each attribute. The last entry in each row denotes the class label. A seperate file which is there which specifies which attributes are discrete and which are continuous. A script is also added to read the data into desired format
### Tasks Done:-
* Construct a decision tree for the above prediction problem. 
* * Preprocessed (before growing the tree) each numerical attribute into a Boolean attribute by a) computing the median value of the attribute in the training data (made sure not to ignore the duplicates) b) replacing each numerical value by a 1/0 value based on whether the value is greater than the median threshold or not. 
* * Repeated for each attribute independently. 
* * For non-Boolean (discrete valued) attributes, used a multi-way split. Used information gain as the criterion for choosing the attribute to split on. 
* * In case of a tie, choosen the attribute which appears first in the ordering as given in the training data. 
* * Ploted the train, validation and test set accuracies against the number of nodes in the tree as you grow the tree. On X-axis ploted the number of nodes in the tree and Y-axis should represent the accuracy.

* Used the scikit-learn library of Python to grow a decision tree. 
* * Tried growing different trees by playing around with parameter values. 
* * Experimented with parameters min samples split, min samples leaf and max depth (varied other parameters as well). 
* * Found the setting of parameters which gives the best accuracy on the validation set. 
* * Reported training, validation and test set accuracies for this parameter setting.

* Next, used the scikit-learn library to learn a random forest over the data. 
* * Tried growing different forests by playing around with parameter values. 
* * Experimented with parameters n estimators, max features and bootstrap (varied other parameters as well). 
* * Found the setting of parameters which gives the best accuracy on the validation set. 
* * Reported training, validation and test set accuracies for this parameter setting.

# Author 
 * [Harshit Monish](https://github.com/harshitmonish)
 
## Course Project Under [Prof. Parag Singla](http://www.cse.iitd.ernet.in/~parags/)
