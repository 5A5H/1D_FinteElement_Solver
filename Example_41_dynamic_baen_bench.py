# TEST for FiniteElement in coupled problems
# for the dynamic terms including inertia and damping 

import numpy as np
import matplotlib.pyplot as plt

import FEM_1D
import FEM_1D.elements.Elmt_BaMo_BaEn_Coupled_1D as ELEMENT


# Create FEM Instance
FEM = FEM_1D.FEM_Simulation(ELEMENT)
FEM.Add_Mesh(100.0,100)
FEM.Add_Material([100,1,1,100,10,0.1],"All")
FEM.Add_EBC("x==0","T",0)
FEM.Add_EBC("x>-1","U",0)
FEM.Add_NBC("x==100","T",1)
FEM.Analysis()

# define a loading function
def load(time):
  lam = 0.0
  if time <= 5:
    lam = (time/5)
  if time > 5:
    lam = 1.0
  if time > 10:
    lam = 0.0
  return lam

#Lets prepare a time loop, with recoding the time and displacement
rec_t = []
rec_u = []
rec_tu = []
nStep, time, dt = 120 ,0.0, 1.0
for step in range(nStep):
  time += dt
  FEM.NextStep(time,load(time))
  print( FEM.NewtonIteration() )
  print( FEM.NewtonIteration() )
  XI = FEM.XI
  TI = FEM.DI[1::2]
  plt.clf()
  plt.plot(XI,TI)
  plt.xlabel('x')
  plt.ylabel('$T$')
  plt.xlim(0,100)
  plt.ylim(-.1,.1)
  plt.pause(.01)
  TEnd = FEM.NodalDof("x==100","T")
