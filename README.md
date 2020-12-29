# OpenVaccine-COVID-19-mRNA-Vaccine-Degradation-Prediction

This repository provides a Pythonic implementation of OpenVaccine-COVID-19-mRNA-Vaccine-Degradation-Prediction by utilizing Python concept of OOP. This repository is is refactored
to fit any regression based problem and consists of the Machine Learning Pipelines (Data Exploratory, Feature Engineering, Cross Validation, Model training and prediction)

Installation
This implementation is written with Python version 3.6 with the listed packages in the requirements.txt file

Clone this repository with git clone https://github.com/Victoloporsche/OpenVaccine-COVID-19-mRNA-Vaccine-Degradation-Prediction.git With Virtual Environent, 
use : a) pip install virtualenv b) cd path-to-the-cloned-repository c) virtualenv myenv d) source myenv/bin/activate 
e) pip install -r requirements.txt With Conda Environment, use: a) cd path-to-the-cloned-repository b) conda create --name myenv c) source activate myenv 
d) pip install -r requirements.txt

# Task Overview (source: https://www.kaggle.com/c/stanford-covid-vaccine/overview)

Winning the fight against the COVID-19 pandemic will require an effective vaccine that can be equitably and widely distributed. Building upon decades of research has allowed scientists to accelerate the search for a vaccine against COVID-19, but every day that goes by without a vaccine has enormous costs for the world nonetheless. We need new, fresh ideas from all corners of the world. Could online gaming and crowdsourcing help solve a worldwide pandemic? Pairing scientific and crowdsourced intelligence could help computational biochemists make measurable progress.

mRNA vaccines have taken the lead as the fastest vaccine candidates for COVID-19, but currently, they face key potential limitations. One of the biggest challenges right now is how to design super stable messenger RNA molecules (mRNA). Conventional vaccines (like your seasonal flu shots) are packaged in disposable syringes and shipped under refrigeration around the world, but that is not currently possible for mRNA vaccines.

Researchers have observed that RNA molecules have the tendency to spontaneously degrade. This is a serious limitation--a single cut can render the mRNA vaccine useless. Currently, little is known on the details of where in the backbone of a given RNA is most prone to being affected. Without this knowledge, current mRNA vaccines against COVID-19 must be prepared and shipped under intense refrigeration, and are unlikely to reach more than a tiny fraction of human beings on the planet unless they can be stabilized.
The Eterna community, led by Professor Rhiju Das, a computational biochemist at Stanford’s School of Medicine, brings together scientists and gamers to solve puzzles and invent medicine. Eterna is an online video game platform that challenges players to solve scientific problems such as mRNA design through puzzles. The solutions are synthesized and experimentally tested at Stanford by researchers to gain new insights about RNA molecules. The Eterna community has previously unlocked new scientific principles, made new diagnostics against deadly diseases, and engaged the world’s most potent intellectual resources for the betterment of the public. The Eterna community has advanced biotechnology through its contribution in over 20 publications, including advances in RNA biotechnology.

In this competition, we are looking to leverage the data science expertise of the Kaggle community to develop models and design rules for RNA degradation. Your model will predict likely degradation rates at each base of an RNA molecule, trained on a subset of an Eterna dataset comprising over 3000 RNA molecules (which span a panoply of sequences and structures) and their degradation rates at each position. We will then score your models on a second generation of RNA sequences that have just been devised by Eterna players for COVID-19 mRNA vaccines. These final test sequences are currently being synthesized and experimentally characterized at Stanford University in parallel to your modeling efforts -- Nature will score your models!

Improving the stability of mRNA vaccines was a problem that was being explored before the pandemic but was expected to take many years to solve. Now, we must solve this deep scientific challenge in months, if not weeks, to accelerate mRNA vaccine research and deliver a refrigerator-stable vaccine against SARS-CoV-2, the virus behind COVID-19. The problem we are trying to solve has eluded academic labs, industry R&D groups, and supercomputers, and so we are turning to you. To help, you can join the team of video game players, scientists, and developers at Eterna to unlock the key in our fight against this devastating pandemic.

Implementation Folder:
The Input files consists of the  mRNA sequence, which requires some cleaning. The dataset can be found here: https://www.kaggle.com/c/stanford-covid-vaccine/data
Model folder consists of the trained model
Output folder consists of the cleaned input data as well as the encoded categorical features
src folder consists of the python and jupyter files
The order of running this repository is:
clean_data.ipynb: This file cleans the input files and outputs a cleaned data which is used for the model 
feature_engineering.py: This class performs feature engineering techniques on the data 
processed_data.py: This class utilizes step 2  
model.py: This class performs cross validation, model training as well as prediction 
main.py: This class combines all the previous classes 
example_openVaccinePredictor.ipynb: This provides the jupyter documentation of the model. 

Next Step: This implementation does not yet consist of optimizing the hyperparameters of the models and for now has an accuracy score of 0.47803. So more modifications 
and commits would be made to this repository from time to time so as to improve this accuracy. Kindly reach out if you have any questions or improvements.

