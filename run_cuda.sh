#!/bin/bash

Pi=3.1415926535897932384626433832795
TimeScheme=1               # 1-Euler, 2-RK4
AdapTime=0                 # 0-dt=dt0, 1-Adpative time step.
InitCond=4                 # 1-Uniform, 2-Randomized, 3-testing sine, 4-a pair of defects solution from Tang et al., 5-a pair of defect solution from sum of isolated defect solution, 6-Load From last Frame
Nx=256
Ny=256
Nb=3                       # Boundary width.
h=0.5                        # Grid step
dt0=0.0001                 # Time step size
Ts=500                    # Final Time. Note that this is not number of time step. 
dte=1.0                    # Time resolution of output
dtQ=0.01                   # Maximum change in Q for calculation of adaptive timestep
ReadSuccess=1              # Used to check if input file is correctly read.
ShowProgress=1 		   # Show current calculation progress
Relax=1                    # Relax initial configuration for InitCond=4
theta0=$Pi                   # Initial angle for uniform distribution or in defect pair
theta1=0                   # Initial angle for second charge in defect pair
charge="-0.5 0.5"          # Charge of defects for InitCond = 4 
pos1="76.5 128.5"
pos2="178.5 128.5"
lambda1=1.5
lambda2=-1
lambda3=1.5
lambda2_p=1
lambda_c=1
lambda_R=1
gamma=1
Gamma=1
A=1
K=1
kappa=10
alpha=0.0
r=2
angle0=-90
angle1=0
dist=100

#for alpha in -0.5 -0.8 -1.0 -1.5 -2.0 -2.5 -3.0 -3.5 -4.0 -4.5 -5.0 -6.0 -7.0 -8.0 -9.0 -10.0 -11.0 -12.0 -13.0 -14.0 -15.0
#for alpha in -0.5 -0.7 -1.0 -1.5 -2.0 -2.5 -3.0 -4.0 -6.0 -8.0 -10.0 -15.0 -20.0 -25.0 -30.0 -40.0
for angle1 in 90
#alpha=0
#for K in 0.2 0.5 2.0 5.0 10.0
do
#angle0=$(bc <<< 'scale=1; '$angle1'-180')
theta0=$(bc <<< 'scale=31; (-0.5*'$angle0')*'$Pi'/180')
theta1=$(bc <<< 'scale=31; (-0.5*'$angle1'-90)*'$Pi'/180')
pos1="$(bc <<< 'scale=1; 128.5-'$dist'/2') 128.5"
pos2="$(bc <<< 'scale=1; 128.5+'$dist'/2') 128.5"
echo $TimeScheme > input.dat
echo $AdapTime >> input.dat
echo $InitCond >> input.dat
echo $Nx >> input.dat
echo $Ny >> input.dat
echo $Nb >> input.dat
echo $h >> input.dat
echo $dt0 >> input.dat
echo $Ts >> input.dat
echo $dte >> input.dat
echo $dtQ >> input.dat
echo $theta0 >> input.dat
echo $theta1 >> input.dat
echo $charge >> input.dat
echo $pos1 >> input.dat
echo $pos2 >> input.dat
echo $lambda1 >> input.dat
echo $lambda2 >> input.dat
echo $lambda3 >> input.dat
echo $lambda2_p >> input.dat
echo $lambda_c >> input.dat
echo $lambda_R >> input.dat
echo $alpha >> input.dat
echo $gamma >> input.dat
echo $Gamma >> input.dat
echo $A >> input.dat
echo $r >> input.dat
echo $K >> input.dat
echo $kappa >> input.dat
echo $ReadSuccess >> input.dat
echo $ShowProgress >> input.dat
echo $Relax >> input.dat
if [ -d "data" ]
then
rm -rf `pwd`/data
fi

mkdir data


./a.out

#mv data data_pp_alpha=$alpha
mv data data_angle=90
#mv data data_angle=$angle1\_alpha=$alpha
done
