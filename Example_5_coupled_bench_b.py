# TEST for FiniteElement in coupled problems
# for the dynamic terms including inertia and damping 

import numpy as np
import matplotlib.pyplot as plt

import FEM_1D
import FEM_1D.elements.Elmt_BaMo_BaEn_Coupled_1D as ELEMENT


# Create FEM Instance
FEM = FEM_1D.FEM_Simulation(ELEMENT)
FEM.Add_Mesh(10.0,10)
FEM.Add_Material([5,1.2,1,10,1e-3,0.4e-4],"All")
FEM.Add_EBC("x==0","U",0)
FEM.Add_EBC("x==0","T",0)
FEM.Add_EBC("x==10","T",10)
FEM.Analysis()
#FEM.verbose = True
# define a loading function
def load(time):
  lam = (time/25)
  if time > 25:
      lam = 1.0
  if time > 50:
    lam = 2.0-((time-25)/25)
  if time > 100:
    lam = -1.0
  return lam

rec_t = []
rec_u = []
rec_T = []
rec_lam = []
nStep, time, dt = 125 ,0.0, 1.0
for step in range(nStep):
  time += dt
  FEM.NextStep(time,load(time))
  FEM.NewtonIteration()
  FEM.NewtonIteration()
  rec_t.append(time)
  rec_T.append( FEM.NodalDof("x==5","T") )
  rec_u.append( FEM.NodalDof("x==10","U") )
  rec_lam.append(load(time))

plt.figure(1,figsize=[20,8])

plt.subplot(131)
plt.plot(rec_t,rec_u)
plt.xlabel('t')
plt.ylabel('$u$')

plt.subplot(132)
plt.plot(rec_t,rec_T)
plt.xlabel('t')
plt.ylabel('$T$')

plt.subplot(133)
plt.plot(rec_t,rec_lam)
plt.xlabel('t')
plt.ylabel('$\lambda$')


plt.show()

