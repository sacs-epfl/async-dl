==============================
Asynchronous algorithms for DL
==============================

We implemented the following asynchronous algorithms for decentralized learning:

* ADPSGD
* Gossip Learning
* Super Gossip Learning (our version of Gossip Learning)

In the tutorial folder, there are different running scripts for each algorithm.
  
**Important**: For ADPSGD experiments, the graph must be bipartite.


-------------------------
Before running the code
-------------------------

Important parameters to set in the config.ini file:

* [SHARING] section ::

    sharing_package = decentralizepy.sharing.AsyncSharing.{ADPSGDSharing, GossipSharing}
    sharing_class = {ADPSGDSharing, GossipSharing}


* [ASYNC_PARAMS] section

  * ``training_time``: the amount of time in minutes that the algorithm should run for
  * ``timeout``: time in seconds, used only in gossip learning
  * ``eval_on_train_set``: if set to True, the train loss is calculated every ``train_evaluate_after`` rounds.
  * ``eval_on_test_set``: if set to True, the model is evaluated on the test set every ``test_after rounds``. Otherwise, the model gets saved in a file.


-------------------------
Evaluation on test set
-------------------------

In the read_and_test folder there is a python script for reading the saved models and evaluating them on the test set of CIFAR10 dataset.
After the execution of read_and_test.py, a json file is generated for each node containing test accuracy and test loss with respect to iterations and time.


