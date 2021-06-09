.. EWSNet documentation master file, created by
   sphinx-quickstart on Sat May 15 13:54:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EWSNet's documentation!
====================================
.. toctree::
   :maxdepth: 2
   :caption: Contents:


This is the official code documentation for the Early Warning Signal Network (EWSNet), a deep learning model trained using simulated stochastic time series data for anticipating 
transitions and regime shifts in complex dynamical systems.


Inference & Finetuning using Pretrained EWSNet
===============================================

.. autoclass:: src.inference.ewsnet.EWSNet
   :members:  predict, finetune ,build_model, load_model

Data Generation and Preprocessing
===================================
.. mat:autofunction:: src.generate_data.model1


Data Loading
=====================
.. automodule:: src.utils.generic_utils
   :members:

Model Training and Evaluation
==============================
.. automodule:: src.model_training.exp_utils
   :members:


Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
