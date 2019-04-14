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

* Reduced overfitting in decision trees bu growing the tree fully and then used post-pruning based on a validation set. 
* * In post-pruning, we greedily prune the nodes of the tree (and sub-tree below them) by iteratively picking a node to prune so that resultant tree gives maximum increase in accuracy on the validation set. In other words, among all the nodes in the tree, we prune the node such that pruning it(and sub-tree below it) results in maximum increase in accuracy over the validation set. 
* * This is repeated until any further pruning leads to decrease in accuracy over the validation set. 

* Used the scikit-learn library of Python to grow a decision tree.Tried growing different trees by playing around with parameter values.

* Used the scikit-learn library to learn a random forest over the data. Tried growing different forests by playing around with parameter values. 
