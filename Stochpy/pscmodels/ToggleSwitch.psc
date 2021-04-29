# Keywords
Description: Genetic Toggle Switch (bistability)
Modelname: ToggleSwitch
Output_In_Conc: False
Species_In_Conc: False
 

# Reactions
R1:
    G1_on > G1_on + P1
    Kp1*G1_on
R2:
    G2_on > G2_on + P2
    Kp2*G2_on
R3:
    P1 + P1 + G2_on > G2_off
    Kb2*G2_on*P1*(P1-1)
R4:
    P2 + P2 + G1_on > G1_off
    Kb1*G1_on*P2*(P2-1)
R5:
    G2_off > P1 + P1 + G2_on
    Ku2*G2_off
R6:
    G1_off > P2 + P2 + G1_on
    Ku1*G1_off
R7:
    P1 > $pool
    Kd1*P1
R8:
    P2 > $pool
    Kd2*P2
 
# Fixed species
 
# Variable species
G1_on = 1.0
G1_off = 0.0
G2_on = 1.0
G2_off = 0.0
P1 = 0.0
P2 = 0.0
 
# Parameters
Kp1 = 1.0
Kd1 = 0.01
Kb1 = 1.0
Ku1 = 1.0
Kp2 = 1.0
Kd2 = 0.01
Kb2 = 1.0
Ku2 = 1.0

