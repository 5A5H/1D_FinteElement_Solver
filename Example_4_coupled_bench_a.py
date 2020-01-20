# TEST for FiniteElement in coupled problems
# for the dynamic terms including inertia and damping 

import numpy as np
import matplotlib.pyplot as plt

import FEM_1D
import FEM_1D.elements.Elmt_BaMo_BaEn_Coupled_1D as ELEMENT


# Create FEM Instance
FEM = FEM_1D.FEM_Simulation(ELEMENT)
FEM.Add_Mesh(10.0,1)
FEM.Add_Material([5,1.2,0,0,1e-2,0],"All")
FEM.Add_EBC("x==0","U",0)
FEM.Add_EBC("x==0","T",0)
FEM.Add_EBC("x==10","T",10)
FEM.Analysis()
FEM.verbose = True
FEM.verbose_system = True

FEM.NextStep(1,1)
print( FEM.NewtonIteration() )
print( FEM.NewtonIteration() )

u = FEM.NodalDof("x==10","U")
print("final disp: ",u)

plt.figure(1,figsize=[20,8])

XI = FEM.XI
UI = FEM.DI[0::2]
plt.subplot(211)
plt.plot(XI,UI)
plt.xlabel('x')
plt.ylabel('$u$')

XI = FEM.XI
TI = FEM.DI[1::2]
plt.subplot(212)
plt.plot(XI,TI)
plt.xlim(0,10)
plt.ylim(-1,11)
plt.xlabel('x')
plt.ylabel('$T$')

plt.show()