Modelname: MAPK
Description: Ultrasensitive MAPK cascade signal with feedback

# Set model to run with numbers of individuals
Species_In_Conc: False
Output_In_Conc: False


# Differential Equations as Reactions
R1:

	M3K > M3Kp
	V1*M3K/( (1+(MAPKpp/Kl)**n)*(K1+M3K) )

R2:
	M3Kp > M3K
	V2*M3Kp/(K2+M3Kp)

R3:	
	M2K > M2Kp
	k3*M3Kp*M2K/(K3+M2K)

R4:
	M2Kp > M2Kpp
	k4*M3Kp*M2Kp/(K4+M2Kp)

R5:
	M2Kpp > M2Kp
	V5*M2Kpp/(K5+M2Kpp)

R6:
	M2Kp > M2K
	V6*M2Kp/(K6+M2Kp)

R7:
	MAPK > MAPKp
	k7*M2Kpp*MAPK/(K7+MAPK)

R8:
	MAPKp > MAPKpp
	k8*M2Kpp*MAPKp/(K8+MAPKp)

R9:
	MAPKpp > MAPKp
	V9*MAPKpp/(K9+MAPKpp)

R10:
	MAPKp > MAPK
	V10*MAPKp/(K10+MAPKp)
	
	
	
# Parameter values
M3K = 50
M3Kp = 50
M2K = 100
M2Kp = 100
M2Kpp = 100
MAPK = 100
MAPKp = 100
MAPKpp = 100

V1 = 0.1 # 2.5
n = 1
Kl = 9
K1 = 10
V2 = 0.25
K2 = 8
k3 = 0.025
K3 = 15
k4 = 0.025
K4 = 15
V5 = 0.75 
K5 = 15
V6 = 0.75 
K6 = 15
k7 = 0.025
K7 = 15
k8 = 0.025
K8 = 15
V9 = 0.5 
K9 = 15
V10 = 0.5 
K10 = 15

#M3Ktot = 100
#M2Ktot = 300
#MAPKtot = 300

#FIX: M3Ktot M2Ktot MAPKtot

# Total concentrations
#!F M3Ktot = M3K+M3Kp
#!F M2Ktot = M2K+M2Kp+M2Kpp
#!F MAPKtot = MAPK+MAPKp+MAPKpp


