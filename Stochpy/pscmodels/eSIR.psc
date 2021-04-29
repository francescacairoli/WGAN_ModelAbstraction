# Susceptible-Infected-Recovered dynamic disease transmission model
# PyCSeS Implementation
# Author: Eric Lofgren (Eric.Lofgren@gmail.com)

Modelname: eSIRS
Description: PySCes Model Description Language Implementation of SIR model

# Set model to run with numbers of individuals
Species_In_Conc: False
Output_In_Conc: False

# Differential Equations as Reactions
R1:
	S > I
	beta*S*I/(S+I+R)+eta*S
	
R2:
	I > R
	gamma*I

R3:
	R > S
	tau*R

# Parameter Values
S = 60
I = 10
R = 30
beta = 3
gamma = 1
tau = 0.5
eta = 0.1

# Total population size, N
!F N = S+I+R 
