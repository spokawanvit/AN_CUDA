#!/bin/bash

Pi=3.1415926535897932384626433832795
TimeScheme=1                      # 1-Euler, 2-RK4
AdapTime=0                        # 0-dt=dt0, 1-Adpative time step.
InitCond=4                        # 1-Uniform, 2-Randomized, 3-testing sine, 4-a pair of defects solution from Tang et al., 5-a pair of defect solution from sum of isolated defect solution, 6-Load From last Frame
Nx=256
Ny=256
Nb=3                              # Boundary width.
h=0.5                             # Grid step
dt0=0.0001                        # Time step size
Ts=10                             # Final Time. Note that this is not number of time step.
dte=1.0                           # Time resolution of output
dtQ=0.01                          # Maximum change in Q for calculation of adaptive timestep
ReadSuccess=1                     # Used to check if input file is correctly read.
ShowProgress=1 		                # Show current calculation progress
Relax=1                           # Relax initial configuration for InitCond=4
theta0=$Pi                        # Initial angle for uniform distribution or in defect pair
theta1=0                          # Initial angle for second charge in defect pair
defnum=4                          # number of defects
charge="-0.5 0.5 0.5 -0.5"        # Charge of defects for InitCond = 4
xcoord="76.5 76.5 178.5 178.5"
ycoord="178.5 76.5 178.5 76.5"
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

for angle1 in 90
do
theta0=$(bc <<< 'scale=31; (-0.5*'$angle0')*'$Pi'/180')
theta1=$(bc <<< 'scale=31; (-0.5*'$angle1'-90)*'$Pi'/180')
pos1="$(bc <<< 'scale=1; 128.5-'$dist'/2') 128.5"
pos2="$(bc <<< 'scale=1; 128.5+'$dist'/2') 128.5"
local_theta="$(bc <<< 'scale=31; 0*'$Pi'/180') $(bc <<< 'scale=31; 0*'$Pi'/180') $(bc <<< 'scale=31; 0*'$Pi'/180') $(bc <<< 'scale=31; 0*'$Pi'/180')"
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
echo $defnum >> input.dat
echo $charge >> defect.dat
echo $xcoord >> defect.dat
echo $ycoord >> defect.dat
echo $local_phase >> defect.dat
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
