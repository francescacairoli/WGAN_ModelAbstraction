# WGAN_ModelAbstraction

## Abstraction of Markov Population Dynamics via Generative Adversarial Nets
Francesca Cairoli, Ginevra Carbone, Luca Bortolussi.

The code containing the experiments and the results presented at the CMSB 21 conference. The folder is organized as follows:

- **Stochpy/psc_models** folder contains the CRN model expressed, as a set of reactions, in the Stochpy format.
- **Dataset_Generation** folder contains the scripts to generate the datasets through SSA simulation (based on Stochpy library):
- **Trajectory_Abstraction** folder contains the implementation of the c-WCGAN-GP with either fixed parameters (*gp_model_abstraction.py*) or with an arbitrary number of varying parameters (*param_gp_model_abstraction.py*).

The command to repeate the experiments prensented in the paper are specified in the file *commands_for_experiments.txt*.

The datasets used are shared at the following link: https://mega.nz/folder/1WZGmJSL#J9Wjzi2zIxesrEAEqJUJqw
