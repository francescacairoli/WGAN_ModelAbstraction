
# Keywords
Description: Clock
Output_In_Conc: False
Species_In_Conc: Falsee
 

# Reactions
R1:
    A+B > A+A
    k*A*B/(A+B+C)
R2:
    B+C > B+B
    k*B*C/(A+B+C)
R3:
    C+A > C+C
    k*C*A/(A+B+C)

 
# Variable species
A = 10
B = 10
C = 10
 
# Parameters
k = 0.5

