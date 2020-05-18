#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cuda_runtime_api.h"
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cufft.h>
#include <cufftXt.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <vector>
#include <string>
#include <time.h>
#include <math.h>

//============================================================================
// Definitions ---------------------------------------------------------------
// Define the precision of real numbers, could be float/double.
#define real double
#define Pi 3.1415926535897932384626433832795

using namespace std;

// Initialize random number seed. To generate a random number, use:
// uniform_real_distribution<real> randUR; a=randUR(rng);
// or:
// uniform_int_distribution<int> randUI; a=randUI(rng);
int seed = time(0);

struct Fields {
  real * Qxx;
  real * Qxy;
  real * dtQxx;
  real * dtQxy;
};

struct DefectAttrs {
  real * charges;
  real * xcoord;
  real * ycoord;
  real * local_phase;
};

struct Parameters {
  int TimeScheme;         // 1-Euler, 2-RK4
  int AdapTime;           // 0-dt=dt0, 1-Adpative time step.
  int InitCond;           // 1-Uniform, 2-Randomized, 3-Test
  int Nx;
  int Ny;
  int Nb;                 // Boundary for pseudo periodic condition
  real h;
  real dt0;               // Timestep size
  real Ts;
  real dte;               // Time resolution of output
  real dtQ;
  real theta0;
  real theta1;
  int defnum;
  real lambda1;
  real lambda2;
  real lambda3;
  real lambda2_p;
  real lambda_c;
  real lambda_R;
  real alpha;
  real gamma;
  real Gamma;
  real A;
  real r;
  real K;
  real kappa;
  int GSize;
  int BSize;
  int DASize;
  int ReadSuccess;
  int STDout_result;
  int Relax;
};


int iStop;
real T;
real Dt;
real progress;
real * t;
real * dt;

// Use F/f to store fields in host/device.
Fields F,f;
// Similarily, we use P/p to store parameters in host/device. And D,d to store defects' positions and charges
Parameters P;
DefectAttrs D,d;
__constant__ Parameters p;
// Declare other global run-time variables.
real * dtQMX;
real * dtQM;
real DtQM;
clock_t tStart;

// Declare functions  --------------------------------------------------------
// Host functions
void initRandSeed();
void GetInput();
void GetDAInput();
void MemAlloc();
void MemFree();
void InitConf();
void RelaxIni();
void evolve();
void ShowProgress();
void ExpoConf(string str_t);
double RandUniform();

// Device functions
__global__ void initRandSeedDevi(unsigned int seed, curandState_t* states);
__global__ void getTempVari(Fields ff);
__global__ void getRHS(Fields ff);
__global__ void updateEuler(Fields ff, real * dt);
__global__ void updateRK4(Fields ff, real * dt);
__global__ void getMaxX(Fields , real * dtRhoMX);
__global__ void getDt(real * dtRhoMX, real * dtRhoM, real * dt);
__global__ void BounPeriF(Fields ff);
__global__ void relaxQ(Fields ff, DefectAttrs dd, real * dt);
__device__ real d1x(real * u, int i, int j);
__device__ real d1y(real * u, int i, int j);
__device__ real dxy(real * u, int i, int j);
__device__ real d2x(real * u, int i, int j);
__device__ real d2y(real * u, int i, int j);
__device__ real d2x2y(real * u, int i, int j);
__device__ real d4x(real * u, int i, int j);
__device__ real d4y(real * u, int i, int j);
__device__ real BiLaO4I(real * u, int i, int j);

//============================================================================
int main() {
  // Get starting time of simulation.
  tStart = clock();

  cudaDeviceReset();
  // Get parameters from file.
  GetInput();
  if (P.ReadSuccess==1) {
    // Add a part that check if old data files exist and delete them.
    initRandSeed();
    MemAlloc();
    GetDAInput();
    InitConf();

    if (P.Relax==1) {
      RelaxIni();
    }

    tStart = clock();

    evolve();
    MemFree();
  }

  clock_t tEnd = clock();
  if (P.STDout_result){
    cout<<"Simulation finished. ";
    cout<<"CPU time = "<<double(tEnd - tStart)/CLOCKS_PER_SEC<<" sec"<<endl;
  }
}

//============================================================================
void evolve() {
  std::string str_t;
  if (P.STDout_result == 1){
    ShowProgress();
  }
  ExpoConf("0");
  while (iStop==0) {

    // Evolution
    if (P.TimeScheme==1) {
      BounPeriF<<<P.Ny,P.Nx>>>(f);
      getRHS<<<P.Ny,P.Nx>>>(f);
      if (P.AdapTime==1) {
      	getMaxX<<<2*P.Ny,P.Nx,P.Nx*sizeof(real)>>>(f, dtQMX);
      	getDt<<<1,2*P.Ny,2*P.Ny*sizeof(real)>>>(dtQMX, dtQM, dt);
      	// cudaMemcpy(&DtRhoM,dtRhoM,sizeof(real),cudaMemcpyDeviceToHost);
      	cudaMemcpy(&Dt,dt,sizeof(real),cudaMemcpyDeviceToHost);
      }
      updateEuler<<<P.Ny,P.Nx>>>(f,dt);
    } else if (P.TimeScheme==2) {
      updateRK4<<<P.Ny,P.Nx>>>(f,dt);
    }
    T=T+Dt;

    // Export.
    if (floor(T/P.dte)>floor((T-Dt)/P.dte)) {
      int te=floor(T/P.dte);
      stringstream ss;
      ss << te;
      string str_t;
      ss >> str_t;
      ExpoConf(str_t);
      if (P.STDout_result == 1){
        ShowProgress();
      }
    }

    // Check.
    if (T>P.Ts) {
      iStop=1;
    }
  }
}

//============================================================================
void RelaxIni(){
  for (int i=0; i<500000; i++){
    BounPeriF<<<P.Ny,P.Nx>>>(f);
    getRHS<<<P.Ny,P.Nx>>>(f);
    relaxQ<<<P.Ny,P.Nx>>>(f,d,dt);
    /*
    if (i%1000 == 0) {
      stringstream ss;
      ss << i/1000;
      string str_t;
      ss >> str_t;
      ExpoConf("Relax_"+str_t);
    }
    */
  }
}

//============================================================================

__global__ void getRHS(Fields ff) { //  Flag
  int i=blockIdx.x;
  int j=threadIdx.x;
  int idx = (blockDim.x+2*p.Nb)*(i+p.Nb)+j+p.Nb;

  // Storing intermediate variables

  real vx;
  real vy;
  real lapl_Qxx;
  real lapl_Qxy;
  vx = p.alpha/p.Gamma * (d1x(ff.Qxx,i,j)+d1y(ff.Qxy,i,j));
  vy = p.alpha/p.Gamma * (d1x(ff.Qxy,i,j)-d1y(ff.Qxx,i,j));
  lapl_Qxx = d2x(ff.Qxx,i,j)+d2y(ff.Qxx,i,j);
  lapl_Qxy = d2x(ff.Qxy,i,j)+d2y(ff.Qxy,i,j);


  if (i%(p.Ny-1) == 0 && j%(p.Nx-1) == 0){
    // Corner grids
    ff.dtQxx[idx] = -p.A/p.gamma*ff.Qxx[idx]*(1.0-p.r+4.0*p.r*(ff.Qxx[idx]*ff.Qxx[idx]+ff.Qxy[idx]*ff.Qxy[idx]));
    ff.dtQxy[idx] = -p.A/p.gamma*ff.Qxy[idx]*(1.0-p.r+4.0*p.r*(ff.Qxx[idx]*ff.Qxx[idx]+ff.Qxy[idx]*ff.Qxy[idx]));

  } else if (i%(p.Ny-1) == 0 && j > 0 && j < p.Nx){
    // top and bottom boundary dy = 0
    ff.dtQxx[idx] = -p.lambda_c*p.alpha/p.Gamma*pow(d1x(ff.Qxx,i,j),2)
    -p.alpha*p.lambda_R/p.Gamma*ff.Qxy[idx]*d2x(ff.Qxy,i,j)
    +p.alpha*p.lambda1/(2.0*p.Gamma)*d2x(ff.Qxx,i,j)-p.alpha*p.lambda3/p.Gamma*ff.Qxx[idx]*(ff.Qxx[idx]*d2x(ff.Qxx,i,j)
      +ff.Qxy[idx]*d2x(ff.Qxy,i,j))+1.0/p.gamma*(-p.A*ff.Qxx[idx]*(1.0-p.r+4.0*p.r*(ff.Qxx[idx]*ff.Qxx[idx]+ff.Qxy[idx]*ff.Qxy[idx]))
      +p.K*d2x(ff.Qxx,i,j)-p.kappa*d4x(ff.Qxx,i,j));

    ff.dtQxy[idx] = -p.lambda_c*p.alpha/p.Gamma *pow(d1x(ff.Qxx,i,j),2)
    +p.alpha*p.lambda_R/p.Gamma*ff.Qxx[idx]*d2x(ff.Qxy,i,j)
    +p.alpha*p.lambda1/(2.0*p.Gamma)*d2x(ff.Qxy,i,j)-p.alpha*p.lambda3/p.Gamma*ff.Qxy[idx]*(ff.Qxx[idx]*d2x(ff.Qxx,i,j)
      +ff.Qxy[idx]*d2x(ff.Qxy,i,j))+1.0/p.gamma*(-p.A*ff.Qxy[idx]*(1.0-p.r+4.0*p.r*(ff.Qxx[idx]*ff.Qxx[idx]+ff.Qxy[idx]*ff.Qxy[idx]))
      +p.K*d2x(ff.Qxy,i,j)-p.kappa*d4x(ff.Qxy,i,j));

  } else if (j%(p.Nx-1) == 0 && i > 0 && i < p.Ny){
    // left and right boundary dx = 0
    ff.dtQxx[idx] = p.lambda_c*p.alpha/p.Gamma *pow(d1y(ff.Qxx,i,j),2);
    +p.alpha*p.lambda_R/p.Gamma*ff.Qxy[idx]*d2y(ff.Qxy,i,j)
    +p.alpha*p.lambda1/(2.0*p.Gamma)*d2y(ff.Qxx,i,j)-p.alpha*p.lambda3/p.Gamma*ff.Qxx[idx]*(ff.Qxx[idx]*d2y(ff.Qxx,i,j)
      +ff.Qxy[idx]*d2y(ff.Qxy,i,j))+1.0/p.gamma*(-p.A*ff.Qxx[idx]*(1.0-p.r+4.0*p.r*(ff.Qxx[idx]*ff.Qxx[idx]+ff.Qxy[idx]*ff.Qxy[idx]))
      +p.K*d2y(ff.Qxx,i,j)-p.kappa*d4y(ff.Qxx,i,j));

    ff.dtQxy[idx] = p.lambda_c*p.alpha/p.Gamma*pow(d1y(ff.Qxy,i,j),2);
    -p.alpha*p.lambda_R/p.Gamma*ff.Qxx[idx]*d2y(ff.Qxy,i,j)
    +p.alpha*p.lambda1/(2.0*p.Gamma)*d2y(ff.Qxy,i,j)-p.alpha*p.lambda3/p.Gamma*ff.Qxy[idx]*(ff.Qxx[idx]*d2y(ff.Qxx,i,j)
      +ff.Qxy[idx]*d2y(ff.Qxy,i,j))+1.0/p.gamma*(-p.A*ff.Qxy[idx]*(1.0-p.r+4.0*p.r*(ff.Qxx[idx]*ff.Qxx[idx]+ff.Qxy[idx]*ff.Qxy[idx]))
      +p.K*d2y(ff.Qxy,i,j)-p.kappa*d4y(ff.Qxy,i,j));

  } else if (i<p.Ny && j<p.Nx && i >0 && j > 0) {
    ff.dtQxx[idx] = -p.lambda_c*(vx*d1x(ff.Qxx,i,j)+vy*d1y(ff.Qxx,i,j))
    +p.alpha*p.lambda_R/p.Gamma*ff.Qxy[idx]*(2.0*dxy(ff.Qxx,i,j)-d2x(ff.Qxy,i,j)+d2y(ff.Qxy,i,j))
    +p.alpha*p.lambda1/(2.0*p.Gamma)*(lapl_Qxx)-p.alpha*p.lambda3/p.Gamma*ff.Qxx[idx]*(ff.Qxx[idx]*lapl_Qxx
      +ff.Qxy[idx]*lapl_Qxy)+1.0/p.gamma*(-p.A*ff.Qxx[idx]*(1.0-p.r+4.0*p.r*(ff.Qxx[idx]*ff.Qxx[idx]+ff.Qxy[idx]*ff.Qxy[idx]))
      +p.K*lapl_Qxx-p.kappa*(d4x(ff.Qxx,i,j)+2*d2x2y(ff.Qxx,i,j)+d4y(ff.Qxx,i,j)));

    ff.dtQxy[idx] = -p.lambda_c*(vx*d1x(ff.Qxy,i,j)+vy*d1y(ff.Qxy,i,j))
    -p.alpha*p.lambda_R/p.Gamma*ff.Qxx[idx]*(2.0*dxy(ff.Qxx,i,j)-d2x(ff.Qxy,i,j)+d2y(ff.Qxy,i,j))
    +p.alpha*p.lambda1/(2.0*p.Gamma)*(lapl_Qxy)-p.alpha*p.lambda3/p.Gamma*ff.Qxy[idx]*(ff.Qxx[idx]*lapl_Qxx
      +ff.Qxy[idx]*lapl_Qxy)+1.0/p.gamma*(-p.A*ff.Qxy[idx]*(1.0-p.r+4.0*p.r*(ff.Qxx[idx]*ff.Qxx[idx]+ff.Qxy[idx]*ff.Qxy[idx]))
      +p.K*lapl_Qxy-p.kappa*(d4x(ff.Qxy,i,j)+2*d2x2y(ff.Qxy,i,j)+d4y(ff.Qxy,i,j)));
  }

/*
  if (i<p.Ny && j<p.Nx) {
    ff.dtQxx[idx] = -p.lambda_c*(vx*d1x(ff.Qxx,i,j)+vy*d1y(ff.Qxx,i,j))
    +p.alpha*p.lambda_R/p.Gamma*ff.Qxy[idx]*(2.0*dxy(ff.Qxx,i,j)-d2x(ff.Qxy,i,j)+d2y(ff.Qxy,i,j))
    +p.alpha*p.lambda1/(2.0*p.Gamma)*(lapl_Qxx)-p.alpha*p.lambda3/p.Gamma*ff.Qxx[idx]*(ff.Qxx[idx]*lapl_Qxx
      +ff.Qxy[idx]*lapl_Qxy)+1.0/p.gamma*(-p.A*ff.Qxx[idx]*(1.0-p.r+4.0*p.r*(ff.Qxx[idx]*ff.Qxx[idx]+ff.Qxy[idx]*ff.Qxy[idx]))
      +p.K*lapl_Qxx-p.kappa*(d4x(ff.Qxx,i,j)+2*d2x2y(ff.Qxx,i,j)+d4y(ff.Qxx,i,j)));

    ff.dtQxy[idx] = -p.lambda_c*(vx*d1x(ff.Qxy,i,j)+vy*d1y(ff.Qxy,i,j))
    -p.alpha*p.lambda_R/p.Gamma*ff.Qxx[idx]*(2.0*dxy(ff.Qxx,i,j)-d2x(ff.Qxy,i,j)+d2y(ff.Qxy,i,j))
    +p.alpha*p.lambda1/(2.0*p.Gamma)*(lapl_Qxy)-p.alpha*p.lambda3/p.Gamma*ff.Qxy[idx]*(ff.Qxx[idx]*lapl_Qxx
      +ff.Qxy[idx]*lapl_Qxy)+1.0/p.gamma*(-p.A*ff.Qxy[idx]*(1.0-p.r+4.0*p.r*(ff.Qxx[idx]*ff.Qxx[idx]+ff.Qxy[idx]*ff.Qxy[idx]))
      +p.K*lapl_Qxy-p.kappa*(d4x(ff.Qxy,i,j)+2*d2x2y(ff.Qxy,i,j)+d4y(ff.Qxy,i,j)));
  }
*/
}

//============================================================================

__global__ void relaxQ(Fields ff, DefectAttrs dd, real * ddt) {
  int i=blockIdx.x;
  int j=threadIdx.x;
  int idx=(blockDim.x+2*p.Nb)*(i+p.Nb)+j+p.Nb;

  bool fixed = false;

  for (int n = 0; n < p.defnum; n++) {
    if (((j == floor(dd.xcoord[n]) || j == ceil(dd.xcoord[n]))&&(i <= ceil(dd.ycoord[n])+1 || i >= floor(dd.ycoord[n])))||
    ((j == floor(dd.xcoord[n])-1||j == ceil(dd.xcoord[n])+1)&&(i == floor(dd.ycoord[n])|| i == ceil(dd.ycoord[n])))) {
      fixed = true;
    }
  }

  if (!fixed) {
    ff.Qxx[idx] += *ddt*ff.dtQxx[idx];
    ff.Qxy[idx] += *ddt*ff.dtQxy[idx];
  }
}

//============================================================================
__global__ void updateEuler(Fields ff, real * ddt) {
  int i=blockIdx.x;
  int j=threadIdx.x;
  int idx=(blockDim.x+2*p.Nb)*(i+p.Nb)+j+p.Nb;

  if (i<p.Ny && j<p.Nx) {
    ff.Qxx[idx] += *ddt*ff.dtQxx[idx];
    ff.Qxy[idx] += *ddt*ff.dtQxy[idx];
  }
}

//============================================================================
__global__ void updateRK4(Fields ff, real * ddt) {
  int i=blockIdx.x;
  int j=threadIdx.x;
  int idx = (blockDim.x+2*p.Nb)*(i+p.Nb)+j+p.Nb;

  if (i<p.Ny && j<p.Nx) {
    ff.Qxx[idx] += *ddt*ff.dtQxx[idx];
    ff.Qxy[idx] += *ddt*ff.dtQxy[idx];
  }
}

//============================================================================
__global__ void getMaxX(Fields ff, real *dtQMX) {
  extern __shared__ real sdata[];
  int i=blockIdx.x;
  int j=threadIdx.x;
  int idxA = (blockDim.x+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int idxB = (blockDim.x+2*p.Nb)*(i-p.Ny+p.Nb)+j+p.Nb;

  if (i<p.Ny) {
    sdata[j] = abs(ff.dtQxx[idxA]);
  } else {
    sdata[j] = abs(ff.dtQxy[idxB]);
  }
  __syncthreads();

  for(unsigned int s=blockDim.x/2 ; s >= 1 ; s=s/2) {
    if(j < s) {
      if(sdata[j] < sdata[j+s]) {
	sdata[j] = sdata[j+s];
      }
    }
    __syncthreads();
  }

  if(j == 0 ) {
    dtQMX[i] = sdata[0];
  }
}

//============================================================================
__global__ void getDt(real * dtQMX, real * dtQM, real * dt) {
  extern __shared__ real sdata[];
  unsigned int i = blockIdx.x;
  unsigned int j = threadIdx.x;
  int idx = blockDim.x*i+j;

  // each thread loads one element from global to shared mem
  sdata[j] = dtQMX[idx];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=1; s < blockDim.x; s *= 2) {
    if (j % (2*s) == 0 && sdata[j]<sdata[j+s]) {
      sdata[j] = sdata[j + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (j == 0) {
    dtQM[0] = sdata[0];
    real adpt_t = p.dtQ/sdata[0];
    if (adpt_t >= 100*p.dt0){
    dt[0] = 100*p.dt0;
    }
    else if (adpt_t <= p.dt0){
    dt[0] = p.dt0;
    }
    else{
    dt[0] = adpt_t;
    }
  }
}

//============================================================================
void GetInput() {
  ifstream InputFile ("input.dat");

  P.ReadSuccess=0;
  InputFile >> P.TimeScheme;
  InputFile >> P.AdapTime;
  InputFile >> P.InitCond;
  InputFile >> P.Nx;
  InputFile >> P.Ny;
  InputFile >> P.Nb;
  InputFile >> P.h;
  InputFile >> P.dt0;
  InputFile >> P.Ts;
  InputFile >> P.dte;
  InputFile >> P.dtQ;
  InputFile >> P.theta0;
  InputFile >> P.theta1;
  InputFile >> P.defnum;
  InputFile >> P.lambda1;
  InputFile >> P.lambda2;
  InputFile >> P.lambda3;
  InputFile >> P.lambda2_p;
  InputFile >> P.lambda_c;
  InputFile >> P.lambda_R;
  InputFile >> P.alpha;
  InputFile >> P.gamma;
  InputFile >> P.Gamma;
  InputFile >> P.A;
  InputFile >> P.r;
  InputFile >> P.K;
  InputFile >> P.kappa;
  InputFile >> P.ReadSuccess;
  InputFile >> P.STDout_result;
  InputFile >> P.Relax;

  // Grid size.
  P.GSize=(P.Nx+2*P.Nb)*(P.Ny+2*P.Nb);
  // Byte size of the fields.
  P.BSize=P.GSize*sizeof(real);
  // Byte size of the defect attributes.
  P.DASize=P.defnum*sizeof(real);

  InputFile.close();
  if (P.ReadSuccess==0) {
    cout << "Error while reading the input file!" << endl;
  }

}

void GetDAInput() {
  // For reading defect files
  ifstream DAInputFile ("defect.dat");

  for (int i=0; i < P.defnum; i++) DAInputFile >> D.charges[i];
  for (int i=0; i < P.defnum; i++) DAInputFile >> D.xcoord[i];
  for (int i=0; i < P.defnum; i++) DAInputFile >> D.ycoord[i];
  for (int i=0; i < P.defnum; i++) DAInputFile >> D.local_phase[i];


}

//============================================================================
void InitConf() {
  t=0;
  iStop=0;
  Dt=P.dt0;
  srand(seed); // setting seed for RNG

  // Copy parameters from host memory to device memory
  cudaMemcpyToSymbol(p,&P,sizeof(Parameters));
  cudaMemcpy(t,&T,sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(dt,&Dt,sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(d.xcoord,D.xcoord,P.DASize,cudaMemcpyHostToDevice);
  cudaMemcpy(d.ycoord,D.ycoord,P.DASize,cudaMemcpyHostToDevice);


  // uniform_real_distribution<real> randUR; a=randUR(rng);
  int idx;
  real theta;
  real S;

  // Assigning initial average S for steady sate solution
  if (P.r <= 1) {
    S = 1; // Should have been 0 but we still need randomized initial condition
  }
  else if (P.r >1) {
    S = sqrt((P.r-1)/P.r);
  }

  if (P.InitCond==1) {
    for (int i=0; i<P.Ny; i++) {
      for (int j=0; j<P.Nx; j++) {
        idx=(P.Nx+2*P.Nb)*(i+P.Nb)+j+P.Nb;
        theta = RandUniform()*2*Pi*0.05; // 5 % random error
        F.Qxx[idx] = 0.5*S*cos(2*P.theta0+2*theta);
        F.Qxy[idx] = 0.5*S*sin(2*P.theta0+2*theta);
      }
    }
  }

  if (P.InitCond==2) {
    for (int i=0; i<P.Ny; i++) {
      for (int j=0; j<P.Nx; j++) {
        idx=(P.Nx+2*P.Nb)*(i+P.Nb)+j+P.Nb;
        theta = RandUniform()*2*Pi;
        F.Qxx[idx] = 0.5*S*cos(2*theta);
        F.Qxy[idx] = 0.5*S*sin(2*theta);
      }
    }
  }

  if (P.InitCond==3) {
    for (int i=0; i<P.Ny; i++) {
      for (int j=0; j<P.Nx; j++) {
        idx=(P.Nx+2*P.Nb)*(i+P.Nb)+j+P.Nb;
        theta = 2*Pi*((double)i/P.Ny+(double)j/P.Nx);
        F.Qxx[idx] = 0.5*S*cos(2*theta);
        F.Qxy[idx] = 0.5*S*sin(2*theta);
      }
    }
  }

/*
  if (P.InitCond==4) {
    for (int i=0; i<P.Ny; i++) {
      for (int j=0; j<P.Nx; j++) {
        idx=(P.Nx+2*P.Nb)*(i+P.Nb)+j+P.Nb;
	double xi = sqrt(P.K/(P.A*(P.r-1)));
	double Smax = sqrt((P.r-1)/P.r);
	double r1 = sqrt(pow(j-D.xcoord[0],2)+pow(i-D.ycoord[0],2));
	double r2 = sqrt(pow(j-D.xcoord[1],2)+pow(i-D.ycoord[1],2));
	double theta1 = Pi; //negative charge
	double theta2 = P.theta0; //positive charge

	double deltaTheta = theta2-theta1+(strength[1]-strength[0])*atan2(D.ycoord[0]-D.ycoord[1],D.xcoord[0]-D.xcoord[1]);
	double TTheta = theta1-strength[1]*atan2(D.ycoord[0]-D.ycoord[1],D.xcoord[0]-D.xcoord[1]);

	theta = strength[0]*atan2(i-D.ycoord[0],j-D.xcoord[0])+strength[1]*atan2(i-D.ycoord[1],j-D.xcoord[1])
		+0.5*deltaTheta*(1+(log(pow(j-D.xcoord[0],2)+pow(i-D.ycoord[0],2))-log(pow(j-D.xcoord[1],2)+pow(i-D.ycoord[1],2)))/(log(pow(D.xcoord[1]-D.xcoord[0],2)+pow(D.ycoord[1]-D.ycoord[0],2))))+TTheta;
	double Sr = Smax*0.5*(r1*sqrt((0.34+0.07*r1*r1)/(1+0.41*pow(r1,2)+0.07*pow(r1,4)))+r2*sqrt((0.34+0.07*r2*r2)/(1+0.41*pow(r2,2)+0.07*pow(r2,4))));

        F.Qxx[idx] = 0.5*Sr*cos(2*theta);
        F.Qxy[idx] = 0.5*Sr*sin(2*theta);
      }
    }
  }
*/

  if (P.InitCond==4) {
    for (int i=0; i<P.Ny; i++) {
      for (int j=0; j<P.Nx; j++) {
        idx=(P.Nx+2*P.Nb)*(i+P.Nb)+j+P.Nb;
        double xi = 1.5*sqrt(P.K/(P.A*(P.r-1)))/P.h;
        double Smax = sqrt((P.r-1)/P.r);
        double r1 = sqrt(pow(j-D.xcoord[0],2)+pow(i-D.ycoord[0],2))/xi;
        double r2 = sqrt(pow(j-D.xcoord[1],2)+pow(i-D.ycoord[1],2))/xi;

        double deltaTheta = P.theta1-P.theta0+(D.charges[1]-D.charges[0])*atan2(D.ycoord[0]-D.ycoord[1],D.xcoord[0]-D.xcoord[1]);
        double TTheta = P.theta0-D.charges[1]*atan2(D.ycoord[0]-D.ycoord[1],D.xcoord[0]-D.xcoord[1]);

        theta = D.charges[0]*atan2(i-D.ycoord[0],j-D.xcoord[0])+D.charges[1]*atan2(i-D.ycoord[1],j-D.xcoord[1])
                +0.5*deltaTheta*(1+(log(pow(j-D.xcoord[0],2)+pow(i-D.ycoord[0],2))-log(pow(j-D.xcoord[1],2)+pow(i-D.ycoord[1],2)))/(log(pow(D.xcoord[1]-D.xcoord[0],2)+pow(D.ycoord[1]-D.ycoord[0],2))))+TTheta;
        double Sr = -Smax+Smax*(r1*sqrt((0.34+0.07*r1*r1)/(1+0.41*pow(r1,2)+0.07*pow(r1,4)))+r2*sqrt((0.34+0.07*r2*r2)/(1+0.41*pow(r2,2)+0.07*pow(r2,4))));

        F.Qxx[idx] = 0.5*Sr*cos(2*theta);
        F.Qxy[idx] = 0.5*Sr*sin(2*theta);
      }
    }
  }

  if (P.InitCond==5) {
    for (int i=0; i<P.Ny; i++) {
      for (int j=0; j<P.Nx; j++) {
        idx=(P.Nx+2*P.Nb)*(i+P.Nb)+j+P.Nb;
        theta = 0;
        double Sr = 0;

        /* For periodic boundary condition
        double Smax = sqrt((P.r-1)/P.r);
        double r1 = sqrt(pow(j-D.xcoord[0],2)+pow(i-D.ycoord[0],2));
        double r2 = sqrt(pow(j-D.xcoord[1],2)+pow(i-D.ycoord[1],2));

        for (int n= -20 ; n < 21; n++){
          for (int m = -20; m < 21; m++){
            for (int defnum = 0; defnum < 2; defnum++){
              theta += D.charges[defnum]*atan2(i-pos_y[defnum]-P.Ny*n,j-pos_x[defnum]-P.Nx*m);
            }
          }
        }
        double Sr = Smax*0.5*(r1*sqrt((0.34+0.07*r1*r1)/(1+0.41*pow(r1,2)+0.07*pow(r1,4)))+r2*sqrt((0.34+0.07*r2*r2)/(1+0.41*pow(r2,2)+0.07*pow(r2,4))));
        F.Qxx[idx] = 0.5*Sr*cos(2*theta);
        F.Qxy[idx] = 0.5*Sr*sin(2*theta);
        */

        for (int n = 0; n < P.defnum; n++) {
          double rn = sqrt(pow(j-D.xcoord[n],2)+pow(i-D.ycoord[n],2));
          double BubbleRadius = 10; // radius for bubble around defect in which we will add a localized constant phase to correct initial angle of defect polarization.
          real LocalizedPhase = D.local_phase[n]/(1+exp(rn-BubbleRadius));
          theta += D.charges[n]*atan2(i-D.ycoord[n],j-D.xcoord[n])+LocalizedPhase;
          Sr += rn*sqrt((0.34+0.07*rn*rn)/(1+0.41*rn*rn+0.07*pow(rn,4)));
        }

        double Smax = sqrt((P.r-1)/P.r);
        Sr *= Smax/P.defnum;
        F.Qxx[idx] = 0.5*Sr*cos(2*theta);
        F.Qxy[idx] = 0.5*Sr*sin(2*theta);
      }
    }
  }

  if (P.InitCond==6) {
    string line;
    ifstream LastFrame ("LastFrame.dat");
    for (int i=0; i<P.Ny; i++) {
      for (int j=0; j<P.Nx; j++) {
        idx=(P.Nx+2*P.Nb)*(i+P.Nb)+j+P.Nb;
        getline (LastFrame,line);
        istringstream iss (line);
        iss >> F.Qxx[idx];
        iss >> F.Qxy[idx];
      }
    }
    LastFrame.close();
  }

  cudaMemcpy(f.Qxx,F.Qxx,P.BSize,cudaMemcpyHostToDevice);
  cudaMemcpy(f.Qxy,F.Qxy,P.BSize,cudaMemcpyHostToDevice);
}

//============================================================================
void ExpoConf(string str_t) {
  ofstream ConfFile;
  cudaMemcpy(F.Qxx,f.Qxx,P.BSize,cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Qxy,f.Qxy,P.BSize,cudaMemcpyDeviceToHost);

  std::string ConfFileName="data/conf_" + str_t + ".dat";
  ConfFile.open ( ConfFileName.c_str() );

  int idx;
  for (int i=0; i<P.Ny; i++) {
    for (int j=0; j<P.Nx; j++) {
      idx=(P.Nx+2*P.Nb)*(i+P.Nb)+j+P.Nb;
      ConfFile<<F.Qxx[idx]<<' ';
      ConfFile<<F.Qxy[idx]<<endl;
    }
  }
  ConfFile.close();
}

//============================================================================
void MemAlloc() {
  // Allocate fields in host memory.
  F.Qxx=new real[P.GSize];
  F.Qxy=new real[P.GSize];
  F.dtQxx=new real[P.GSize];
  F.dtQxy=new real[P.GSize];
  D.charges=new real[P.defnum];
  D.xcoord=new real[P.defnum];
  D.ycoord=new real[P.defnum];
  D.local_phase=new real[P.defnum];

  // Allocate memory of fields in device.
  cudaMalloc((void **)&f.Qxx, P.BSize);
  cudaMalloc((void **)&f.Qxy, P.BSize);
  cudaMalloc((void **)&f.dtQxx, P.BSize);
  cudaMalloc((void **)&f.dtQxy, P.BSize);
  cudaMalloc((void **)&d.xcoord, P.DASize);
  cudaMalloc((void **)&d.ycoord, P.DASize);

  cudaMalloc((void **)&dtQMX, 2*P.Ny*sizeof(real));
  cudaMalloc((void **)&dtQM, sizeof(real));
  cudaMalloc((void **)&t, sizeof(real));
  cudaMalloc((void **)&dt, sizeof(real));
}

//============================================================================
void MemFree() {
  // Free host memory
  delete [] F.Qxx;
  delete [] F.Qxy;
  delete [] F.dtQxx;
  delete [] F.dtQxy;
  delete [] D.charges;
  delete [] D.xcoord;
  delete [] D.ycoord;
  delete [] D.local_phase;

  // Free device memory
  cudaFree(f.Qxx);
  cudaFree(f.Qxy);
  cudaFree(f.dtQxx);
  cudaFree(f.dtQxy);
  cudaFree(dtQMX);
  cudaFree(dtQM);
  cudaFree(d.xcoord);
  cudaFree(d.ycoord);
}

//============================================================================

__global__ void BounPeriF(Fields ff)
{
  int i=blockIdx.x;
  int j=threadIdx.x;
  int idx=(p.Nx+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int dj=p.Nx+2*p.Nb;
  int idx1;

  if (i<p.Nb){
    idx1 = idx-(2*i+1)*dj;
    ff.Qxx[idx1] = ff.Qxx[idx];
    ff.Qxy[idx1] = ff.Qxy[idx];
  } else if (i>p.Ny-1-p.Nb){
    idx1 = idx+(2*(p.Ny-i)-1)*dj;
    ff.Qxx[idx1] = ff.Qxx[idx];
    ff.Qxy[idx1] = ff.Qxy[idx];
  }

  if (j<p.Nb){
    idx1 = idx-(2*j+1);
    ff.Qxx[idx1] = ff.Qxx[idx];
    ff.Qxy[idx1] = ff.Qxy[idx];
  } else if (j>p.Nx-1-p.Nb){
    idx1 = idx+(2*(p.Nx-j)-1);
    ff.Qxx[idx1] = ff.Qxx[idx];
    ff.Qxy[idx1] = ff.Qxy[idx];
  }
}
/*
__global__ void BounPeriF(Fields ff)
{
  int i=blockIdx.x;
  int j=threadIdx.x;
  int idx=(p.Nx+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int dj=p.Nx+2*p.Nb;
  int idx1;

  if (i<p.Nb) {
    idx1=idx+p.Ny*dj;
    ff.Qxx[idx1]=ff.Qxx[idx];
    ff.Qxy[idx1]=ff.Qxy[idx];
  } else if (i>p.Ny-1-p.Nb) {
    idx1=idx-p.Ny*dj;
    ff.Qxx[idx1]=ff.Qxx[idx];
    ff.Qxy[idx1]=ff.Qxy[idx];
  }

  if (j<p.Nb) {
    idx1=idx+p.Nx;
    ff.Qxx[idx1]=ff.Qxx[idx];
    ff.Qxy[idx1]=ff.Qxy[idx];
    // Corner grids.
    if (i<p.Nb) {
      idx1=idx+(p.Nx+2*p.Nb)*p.Ny+p.Nx;
      ff.Qxx[idx1]=ff.Qxx[idx];
      ff.Qxy[idx1]=ff.Qxy[idx];
    } else if (i>p.Ny-1-p.Nb) {
      idx1=idx-(p.Nx+2*p.Nb)*p.Ny+p.Nx;
      ff.Qxx[idx1]=ff.Qxx[idx];
      ff.Qxy[idx1]=ff.Qxy[idx];
    }
  } else if (j>p.Nx-1-p.Nb) {
    idx1=idx-p.Nx;
    ff.Qxx[idx1]=ff.Qxx[idx];
    ff.Qxy[idx1]=ff.Qxy[idx];
    // Corner grids.
    if (i<p.Nb) {
      idx1=idx+(p.Nx+2*p.Nb)*p.Ny-p.Nx;
      ff.Qxx[idx1]=ff.Qxx[idx];
      ff.Qxy[idx1]=ff.Qxy[idx];
    } else if (i>p.Ny-1-p.Nb) {
      idx1=idx-(p.Nx+2*p.Nb)*p.Ny-p.Nx;
      ff.Qxx[idx1]=ff.Qxx[idx];
      ff.Qxy[idx1]=ff.Qxy[idx];
    }
  }
}
*/


//============================================================================
__device__ real d1x(real * u, int i, int j)
{

  int idx = (p.Nx+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int didx = 1;
  return 1.0/(2.0*p.h)*(u[idx+didx]-u[idx-didx]);

}

//============================================================================
__device__ real d1y(real * u, int i, int j)
{

  int idx=(p.Nx+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int didx=p.Nx+2*p.Nb;
  return 1.0/(2.0*p.h)*(u[idx+didx]-u[idx-didx]);

}

//============================================================================
__device__ real d2x(real * u, int i, int j)
{

  int idx=(p.Nx+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int didx=1;
  return 1.0/(p.h*p.h)*(u[idx+didx]-2*u[idx]+u[idx-didx]);
}

//============================================================================
__device__ real d2y(real * u, int i, int j)
{

  int idx=(p.Nx+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int didx=p.Nx+2*p.Nb;
  return 1.0/(p.h*p.h)*(u[idx+didx]-2*u[idx]+u[idx-didx]);

}

//============================================================================
__device__ real dxy(real * u, int i, int j)
{

  int idx=(p.Nx+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int di=1;
  int dj=p.Nx+2*p.Nb;
  return 1.0/(4.0*p.h*p.h)*(u[idx+di+dj]-u[idx-di+dj]-u[idx+di-dj]+u[idx-di-dj]);

}

//============================================================================
__device__ real d4x(real * u, int i, int j)
{
  int idx=(p.Nx+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int didx=1;
  return 1.0/(pow(p.h,4))*(u[idx+2*didx]-4*u[idx+didx]+6*u[idx]-4*u[idx-didx]+u[idx-2*didx]);
}

//============================================================================
__device__ real d4y(real * u, int i, int j)
{
  int idx=(p.Nx+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int didx=p.Nx+2*p.Nb;
  return 1.0/(pow(p.h,4))*(u[idx+2*didx]-4*u[idx+didx]+6*u[idx]-4*u[idx-didx]+u[idx-2*didx]);
}

//============================================================================
__device__ real d2x2y(real * u, int i, int j)
{
  int idx=(p.Nx+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int di=1;
  int dj=p.Nx+2*p.Nb;
  return 1.0/(pow(p.h,4))*(u[idx+di+dj]-2*u[idx+dj]+u[idx-di+dj]-2*u[idx+di]+4*u[idx]-2*u[idx-di]+u[idx+di-dj]-2*u[idx-dj]+u[idx-di-dj]);
}


//============================================================================
__device__ real BiLaO4I(real * u, int i, int j) {
  int idx=(p.Nx+2*p.Nb)*(i+p.Nb)+j+p.Nb;
  int dj=1;
  int di=p.Nx+2*p.Nb;
  return 1.0/pow(p.h,4)*( 779.0/45.0*u[idx]
    -191.0/45.0*( u[idx+di] + u[idx-di] + u[idx+dj] + u[idx-dj] )
    -187.0/90.0*( u[idx+di+dj] + u[idx-di+dj] + u[idx+di-dj] + u[idx-di-dj] )
    +7.0/30.0*( u[idx+2*di] + u[idx-2*di] + u[idx+2*dj] + u[idx-2*dj] )
    +47.0/45.0*( u[idx+di+2*dj] + u[idx-di+2*dj] + u[idx+di-2*dj] + u[idx-di-2*dj]
      + u[idx+2*di+dj] + u[idx-2*di+dj] + u[idx+2*di-dj] + u[idx-2*di-dj] )
    -29.0/180.0*( u[idx+2*di+2*dj] + u[idx-2*di+2*dj] + u[idx+2*di-2*dj] + u[idx-2*di-2*dj] )
    +1.0/45.0*( u[idx+3*di] + u[idx-3*di] + u[idx+3*dj] + u[idx-3*dj] )
    -17.0/180.0*( u[idx+di+3*dj] + u[idx-di+3*dj] + u[idx+di-3*dj] + u[idx-di-3*dj]
      + u[idx+3*di+dj] + u[idx-3*di+dj] + u[idx+3*di-dj] + u[idx-3*di-dj] )
  );
}


//============================================================================
void initRandSeed () {
  // Initialize random seed in device.
  curandState_t* states;
  cudaMalloc((void**) &states, P.GSize*sizeof(curandState_t));
  initRandSeedDevi<<<P.Ny,P.Nx>>>(time(NULL), states);
}

//============================================================================
__global__ void initRandSeedDevi (unsigned int seed, curandState_t* states) {
  int index = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(seed,index,0,&states[index]);
}

//============================================================================
void ShowProgress() {
  // Print progress.
  progress=T/P.Ts;
  int barWidth = 50;
  clock_t tNow = clock();
  double tUsed=double(tNow-tStart)/CLOCKS_PER_SEC;

  std::cout << "Progress: ";
  std::cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos) std::cout << "=";
    else if (i == pos) std::cout << ">";
    else std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %";
  if (T==0) {
    std::cout <<"\r";
  } else {
    std::cout << ".  " << floor(tUsed/progress*(1-progress)) << "s remains.\r";
  }
  std::cout.flush();
}

//============================================================================
double RandUniform() {
	return (double)rand()/RAND_MAX;
}
