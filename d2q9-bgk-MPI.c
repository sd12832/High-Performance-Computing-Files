/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>
#include "mpi.h"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct {
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  double density;       /* density per link */
  double accel;         /* density redistribution */
  double omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct {
  double speeds[NSPEEDS];
} t_speed;

enum boolean { FALSE, TRUE };

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile, t_param* params, 
                t_speed** cells_ptr, int** obstacles_ptr, double** av_vels_ptr);

/* 
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int accelerate_flow(const t_param params, t_speed* u, int* w, int n);
int propagate(const t_param params, t_speed* u, t_speed* tmp_ucells, int n);
int rebound(const t_param params, t_speed* u, t_speed* tmp_ucells, int* w, int n);
int collision(const t_param params, t_speed* u, t_speed* tmp_ucells, int* w, int n);
int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, int** obstacles_ptr, double** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells);

/* compute average velocity */
double av_velocity(const t_param params, t_speed* u, int* w, int n);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char *file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  double*  av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */
  int      ii, jj, kk;                  /* generic counter */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic,toc;               /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */
  int size, rank, right, left;
  t_speed *u = NULL;
  int *w = NULL;
  t_speed *sendbuf = NULL;
  t_speed *recvbuf = NULL;
  double *sendbuf2 = NULL;
  double *recvbuf2 = NULL;     
  t_speed *tmp_ucells = NULL;  
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  int    tot_cells = 0; 
  double d;

  /* parse the command line */
  if(argc != 3) {
    usage(argv[0]);
  }
  else{
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &obstacles, &av_vels);

  for(ii=0;ii<(params.ny*params.nx);ii++) {
      if(!obstacles[ii]) {
        tot_cells++; 
      }
  }  

  /*Test 0
  printf("%d", tot_cells); 
  */

  MPI_Init( &argc, &argv );
  /* getting size and rank */ 
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  int n; 

  //number of rows in a rank
  if (params.ny == 500) {
    if (rank == 0) {
      n = floor(params.ny/size) + (500% size); 
    } else {
      n = floor(params.ny/size);
    }
  } else {
    n = ceil(params.ny/size);
  }

  right = (rank + 1) % size;
  left = (rank == 0) ? (rank + size - 1) : (rank - 1);

  //length of the graph
  int length = params.nx; 

  //double to hold the length of rank structure 
  double size_u = (n+2)*params.nx;
  // Cell to hold values in each rank 
  u = (t_speed*)malloc(sizeof(t_speed) * (size_u));

  /*
  for(jj=0; jj<((params.nx*params.ny)+(params.nx*size*2)); jj++) {
    for(kk=0; kk< 9; kk++) {
      if (isnan(u[jj].speeds[kk])){
        printf("initialization NAN"); 
        break; 
      }
    }
  }
  */

  w = (int*)malloc(sizeof(int) * (size_u));
  //sendbuf = (t_speed*)malloc(sizeof(t_speed) * (params.nx));
  //recvbuf = (t_speed*)malloc(sizeof(t_speed) * (params.nx));
  sendbuf2 = (double*)malloc(sizeof(double) * (3)*params.nx);
  recvbuf2 = (double*)malloc(sizeof(double) * (3)*params.nx);
  tmp_ucells = (t_speed*)malloc(sizeof(t_speed) * (size_u));

  /*
    if( rank == 0 ){
      for (kk = 0; kk < 9; kk ++) {
        printf("%f",u[params.nx].speeds[kk]);
      }
      printf("\n");
    }
  */

    //writing to the structure within our rank 
  //Accesses non buffer rows
  for (jj=1;jj<(n+1);jj++) {
    //Goes through the cells within each row 
    for (kk=0;kk<length;kk++) {
      u[(jj*length)+kk].speeds[0] = cells[(((rank*n)+jj-1)*length)+kk].speeds[0];
      u[(jj*length)+kk].speeds[1] = cells[(((rank*n)+jj-1)*length)+kk].speeds[1]; 
      u[(jj*length)+kk].speeds[2] = cells[(((rank*n)+jj-1)*length)+kk].speeds[2]; 
      u[(jj*length)+kk].speeds[3] = cells[(((rank*n)+jj-1)*length)+kk].speeds[3]; 
      u[(jj*length)+kk].speeds[4] = cells[(((rank*n)+jj-1)*length)+kk].speeds[4]; 
      u[(jj*length)+kk].speeds[5] = cells[(((rank*n)+jj-1)*length)+kk].speeds[5]; 
      u[(jj*length)+kk].speeds[6] = cells[(((rank*n)+jj-1)*length)+kk].speeds[6]; 
      u[(jj*length)+kk].speeds[7] = cells[(((rank*n)+jj-1)*length)+kk].speeds[7]; 
      u[(jj*length)+kk].speeds[8] = cells[(((rank*n)+jj-1)*length)+kk].speeds[8];  
    }
  }

  

  for (jj=1;jj<(n+1);jj++) {
    //Goes through the cells within each row 
    for (kk=0;kk<length;kk++) {
      w[(jj*length)+kk] = obstacles[(((rank*n)+jj-1)*length)+kk]; 
    }
  }

  /*Test 1
  if (rank == 0) {
    for (jj=0;jj<9;jj++) {
      for (kk=0;kk<9;kk++){
        printf("%f =",u[params.nx+jj].speeds[kk]);
        printf("%f\n",cells[jj].speeds[kk]);
      }
    }
  }
  */

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr,NULL);
  tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);

  for (ii=0;ii<params.maxIters;ii++) {
    /*
    if (rank == 3 && ii>10 && ii < 20){
        printf("prerank %f\n",u[(n-1)*params.nx + 2].speeds[2]);
    } 
    */
    d = 0; 
    //for all the different rows in all parts of the programs
    //Access the first buffer row
    // for (kk=0;kk<params.nx;kk++) {
    //   sendbuf[kk] = u[params.nx+kk]; 
    // }
    // //Send recieve to the left 
    // MPI_Sendrecv(sendbuf, params.nx, MPI_DOUBLE, left, tag,
    // recvbuf, params.nx, MPI_DOUBLE, right, tag,
    // MPI_COMM_WORLD, &status);
    // //Write on the the right 
    // for(kk=0;kk<length;kk++) {
    //   u[(length*(n+1))+kk] = recvbuf[kk];
    // }
    /*Test 2
    if(ii==0){
      for(jj=0; jj<((params.nx*params.ny)+(params.nx*size*2)); jj++) {
        for(kk=0; kk< 9; kk++) {
          if (isnan(u[jj].speeds[kk])) {
            printf("grrr");
            printf("first send %d NAN", ii );
            break; 
          }
        }
      }
    }

    if (rank == 3){
      for (kk=0;kk<params.nx;kk++) {
        printf("%f =",u[params.nx+kk].speeds[3]);
      }
    }
    if (rank == 2){
      for (kk=0;kk<params.nx;kk++) {
        printf("%f\n",u[(length*(n+1))+kk]);  
      }
    }
    */

    // for(kk=0;kk<length;kk++){
    //   sendbuf[kk] = u[(length*(n))+kk];
    // }
    // MPI_Sendrecv(sendbuf, params.nx, MPI_DOUBLE, right, tag,
    // recvbuf, params.nx, MPI_DOUBLE, left, tag,
    // MPI_COMM_WORLD, &status);
    // for(kk=0;kk<params.nx;kk++){
    //   u[kk] = recvbuf[kk];
    // }
    /*
    if(ii==0){
      for(jj=0; jj<((params.nx*params.ny)+(params.nx*size*2)); jj++) {
        for(kk=0; kk< 9; kk++) {
          if (isnan(u[jj].speeds[kk])) {printf("second send %d NAN", ii);
            break; 
          }
        }
      }
    }

    double *sendbuff;
    sendbuff = (double*)(u[0]);


    if (rank == 3){
      for (kk=0;kk<params.nx;kk++) {
        printf("%f =",u[(length*(n))+kk].speeds[4]); 
      }
    }

    if (rank == 4){
      for (kk=0;kk<params.nx;kk++) {
        printf("%f\n",u[kk].speeds[4]);  
      }
    }
    */

    accelerate_flow(params,u,w, n);
    /*test 3.1 
    if (rank == 3 && ii>10 && ii < 20){
        printf("accelerate %f\n",u[(n-1)*params.nx + 2].speeds[2]);
    }

    if(ii==0){
      for(jj=0; jj<((params.nx*params.ny)+(params.nx*size*2)); jj++) {
        for(kk=0; kk< 9; kk++) {
          if (isnan(u[jj].speeds[kk])) {printf("accel NAN");
            break; 
          }
        }
      }
    }
    */
    propagate(params,u,tmp_ucells, n);

  // if (rank == 0)
  // {
  //   // for(jj=0; jj<(n+2); jj++) 
  //   // {
  //     for(kk=0; kk<params.nx; kk++) 
  //     {
  //        // printf("new tmp_cells speeds value 0 at %d: %lf\n", kk, tmp_ucells[kk].speeds[0]);
  //        // printf("new tmp_cells speeds value 1 at %d: %lf\n", kk, tmp_ucells[kk].speeds[1]);
  //        // printf("new tmp_cells speeds value 2 at %d: %lf\n", kk, tmp_ucells[kk].speeds[2]);
  //        // printf("new tmp_cells speeds value 3 at %d: %lf\n", kk, tmp_ucells[kk].speeds[3]);
  //         printf("tmp_cells speeds value 6 at %d: %lf\n", kk, tmp_ucells[length*(n)+kk].speeds[6]);
  //        // printf("new tmp_cells speeds value 5 at %d: %lf\n", kk, tmp_ucells[kk].speeds[5]);
  //        // printf("new tmp_cells speeds value 6 at %d: %lf\n", kk, tmp_ucells[kk].speeds[6]);
  //         printf("tmp_cells speeds value 2 at %d: %lf\n", kk, tmp_ucells[length*(n)+kk].speeds[2]);
  //         printf("tmp_cells speeds value 5 at %d: %lf\n", kk, tmp_ucells[length*(n)+kk].speeds[5]);
  //     }
  //   // }
  // }

    //The second halo exchange after propagate
    int a = 7;
    int b = 4;
    int c = 8; 
    for (kk=0;kk<params.nx;kk++) {

      sendbuf2[3*kk]   = tmp_ucells[kk].speeds[a]; 
      sendbuf2[3*kk+1] = tmp_ucells[kk].speeds[b]; 
      sendbuf2[3*kk+2] = tmp_ucells[kk].speeds[c]; 
    }
    //Send recieve to the left 
    MPI_Sendrecv(sendbuf2, 3*params.nx, MPI_DOUBLE, left, tag,
    recvbuf2, 3*params.nx, MPI_DOUBLE, right, tag,
    MPI_COMM_WORLD, &status);
    //Write on the the right 
    for(kk=0;kk<length;kk++) {
      tmp_ucells[(length*(n))+kk].speeds[a] = recvbuf2[3*kk];
      tmp_ucells[(length*(n))+kk].speeds[b] = recvbuf2[3*kk+1];
      tmp_ucells[(length*(n))+kk].speeds[c] = recvbuf2[3*kk+2];
    }


    a = 6;
    b = 2;
    c = 5;
    
    for (kk=0;kk<params.nx;kk++) 
    {
      sendbuf2[3*kk]   = tmp_ucells[(n+1)*params.nx + kk].speeds[a]; 
      sendbuf2[3*kk+1] = tmp_ucells[(n+1)*params.nx + kk].speeds[b]; 
      sendbuf2[3*kk+2] = tmp_ucells[(n+1)*params.nx + kk].speeds[c]; 
    }

    //Send recieve to the left 
    MPI_Sendrecv(sendbuf2, 3*params.nx, MPI_DOUBLE, right, tag,
    recvbuf2, 3*params.nx, MPI_DOUBLE, left, tag,
    MPI_COMM_WORLD, &status);
    //Write on the the right 
    for(kk=0;kk<length;kk++)
     {
      // printf("first recieved value 6: %lf\n", recvbuf2[3*kk]);
      // printf("second recieved value 2 : %lf\n", recvbuf2[3*kk+1]);
      // printf("third recieved value 5: %lf\n", recvbuf2[3*kk+2]);
      tmp_ucells[length+kk].speeds[a] = recvbuf2[3*kk];
      tmp_ucells[length+kk].speeds[b] = recvbuf2[3*kk+1];
      tmp_ucells[length+kk].speeds[c] = recvbuf2[3*kk+2];
      //printf("new tmp_cells row 1 speeds value 0 at %d: %lf\n", kk, tmp_ucells[length + kk].speeds[a]);
    } 

    // if (rank == 0)
    // {
    //   for(kk=0; kk<params.nx; kk++) 
    //   {
    //      // printf("new tmp_cells speeds value 0 at %d: %lf\n", kk, tmp_ucells[kk].speeds[0]);
    //      // printf("new tmp_cells speeds value 1 at %d: %lf\n", kk, tmp_ucells[kk].speeds[1]);
    //      // printf("new tmp_cells speeds value 2 at %d: %lf\n", kk, tmp_ucells[kk].speeds[2]);
    //      // printf("new tmp_cells speeds value 3 at %d: %lf\n", kk, tmp_ucells[kk].speeds[3]);
    //       printf("new tmp_cells speeds value 6 at %d: %lf\n", kk, tmp_ucells[length+kk].speeds[6]);
    //      // printf("new tmp_cells speeds value 5 at %d: %lf\n", kk, tmp_ucells[kk].speeds[5]);
    //      // printf("new tmp_cells speeds value 6 at %d: %lf\n", kk, tmp_ucells[kk].speeds[6]);
    //       printf("new tmp_cells speeds value 2 at %d: %lf\n", kk, tmp_ucells[length+kk].speeds[2]);
    //       printf("new tmp_cells speeds value 5 at %d: %lf\n", kk, tmp_ucells[length+kk].speeds[5]);
    //   }
    // }
  // if (rank == 0)
  // {
  //   // for(jj=0; jj<(n+2); jj++) 
  //   // {
  //     for(kk=0; kk<params.nx; kk++) 
  //     {
  //        printf("new tmp_cells row 1 speeds value 0 at %d: %lf\n", kk, tmp_ucells[length + kk].speeds[0]);
  //        printf("new tmp_cells row 1 speeds value 1 at %d: %lf\n", kk, tmp_ucells[length + kk].speeds[1]);
  //        printf("new tmp_cells row 1 speeds value 2 at %d: %lf\n", kk, tmp_ucells[length + kk].speeds[2]);
  //        printf("new tmp_cells row 1 speeds value 3 at %d: %lf\n", kk, tmp_ucells[length + kk].speeds[3]);
  //        printf("new tmp_cells row 1 speeds value 4 at %d: %lf\n", kk, tmp_ucells[length + kk].speeds[4]);
  //        printf("new tmp_cells row 1 speeds value 5 at %d: %lf\n", kk, tmp_ucells[length + kk].speeds[5]);
  //        printf("new tmp_cells row 1 speeds value 6 at %d: %lf\n", kk, tmp_ucells[length + kk].speeds[6]);
  //        printf("new tmp_cells row 1 speeds value 7 at %d: %lf\n", kk, tmp_ucells[length + kk].speeds[7]);
  //        printf("new tmp_cells row 1 speeds value 8 at %d: %lf\n", kk, tmp_ucells[length + kk].speeds[8]);
  //     }
  //   // }
  // }


    


    rebound(params,u,tmp_ucells,w, n);
    /*test 3.3
    if (rank == 3 && ii>10 && ii < 20) {
        printf("rebound %f\n",u[(n-1)*params.nx + 2].speeds[2]);
    }
    
    if(ii==0){
      for(jj=0; jj<((params.nx*params.ny)+(params.nx*size*2)); jj++) {
        for(kk=0; kk< 9; kk++) {
          if (isnan(u[jj].speeds[kk])) {printf("rebound NAN");
            break; 
          }
        }
      }
    }
    */

    collision(params,u,tmp_ucells,w, n); 
    /*test 3.4 
    if (rank == 3 && ii>10 && ii < 20){
        printf("Collision %f\n",u[(n-1)*params.nx + 2].speeds[2]);
    }
    if(ii==0){
      for(jj=0; jj<((params.nx*params.ny)+(params.nx*size*2)); jj++) {
        for(kk=0; kk< 9; kk++) {
          if (isnan(u[jj].speeds[kk])) {printf("coll NAN");
            break; 
          }
        }
      }
    }
    */ 

    double rec_tot_u;

    d = av_velocity(params,u,w, n);

    if (rank != 0) {
      MPI_Send(&d, 1 , MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    } else {
      for(jj=1;jj<size;jj++)
      {
        MPI_Recv(&rec_tot_u, 1, MPI_DOUBLE,jj,tag,MPI_COMM_WORLD, &status);
        d += rec_tot_u; 
      }
    }
     av_vels[ii] = d/(double)tot_cells;
    

#ifdef DEBUG
    printf("==timestep: %d==\n",ii);
    printf("av velocity: %.12E\n", av_vels[ii]);
    printf("tot density: %.12E\n",total_density(params,cells));
#endif
  }

  gettimeofday(&timstr,NULL);
  toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr=ru.ru_utime;        
  usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
  timstr=ru.ru_stime;        
  systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);

  /*Write everything back to cells
    if( rank == 0 ){
      for (kk = 0; kk < 9; kk ++) {
        printf("%f", cells[params.nx].speeds[kk]);
      }


    printf("\n"); 
    }
  
  for(jj=0; jj<((params.nx*params.ny)+(params.nx*size*2)); jj++) {
    for(kk=0; kk< 9; kk++) {
      if (isnan(u[jj].speeds[kk])){printf("initialization last");
      break; 
    }
    }
  }
  */
  
  for(ii=1;ii<(n+1);ii++) {
    for(jj=0;jj<params.nx;jj++) {
      cells[((ii-1)*params.nx)+jj] = u[(ii*params.nx)+jj];
    }
  }

  double *send = (double*)(&cells[0]);
  double *recv;

  if (rank != 0)
  {
    MPI_Send(send,params.nx*n*NSPEEDS,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
  }

  else
  {
    for (ii = 1; ii < size; ++ii)
    {
      recv = (double*)(&cells[ii*params.nx*n]);
      MPI_Recv(recv,params.nx*n*NSPEEDS,MPI_DOUBLE,ii,tag,MPI_COMM_WORLD, &status);
    }
  }

  /*Write everything back to cells
    if(rank == 0) for (kk = 0; kk < 9; kk ++) {
        printf("%f", cells[params.nx].speeds[kk]);
    }

  Test 4
  if (rank == 0) {
    for (jj=0;jj<9;jj++) {
      for (kk=0;kk<9;kk++){
        printf("%f =",u[params.nx+jj].speeds[kk]);
        printf("%f\n",cells[jj].speeds[kk]);
      }
    }
  }
  */

  free(u);
  free(w);
  free(tmp_ucells); 
  free(sendbuf);
  free(recvbuf);
  if (rank == 0) {
    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n",calc_reynolds(params,cells,obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc-tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params,cells,obstacles,av_vels);
  }
  finalise(&params, &cells, &obstacles, &av_vels);
  MPI_Finalize();  
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* u, int* w, int n)
{
  int size, rank; 

    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if (rank == size-1) {
      int jj;     /* generic counters */
      double w1,w2;  /* weighting factors */

      /* compute weighting factors */
      w1 = params.density * params.accel / 9.0;
      w2 = params.density * params.accel / 36.0;
      int access_row2 = params.nx*(n-1);
      /* modify the 2nd row of the grid */
      for(jj=0;jj<params.nx;jj++) {
        /* if the cell is not occupied and
        ** we don't send a density negative */
        if( !w[access_row2 + jj] && 
    	(u[access_row2 + jj].speeds[3] - w1) > 0.0 &&
    	(u[access_row2 + jj].speeds[6] - w2) > 0.0 &&
    	(u[access_row2 + jj].speeds[7] - w2) > 0.0 ) {
          /* increase 'east-side' densities */
          u[access_row2 + jj].speeds[1] += w1;
          u[access_row2 + jj].speeds[5] += w2;
          u[access_row2 + jj].speeds[8] += w2;
          /* decrease 'west-side' densities */
          u[access_row2 + jj].speeds[3] -= w1;
          u[access_row2 + jj].speeds[6] -= w2;
          u[access_row2 + jj].speeds[7] -= w2;
        }
      }
    }
  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* u, t_speed* tmp_ucells, int n)
{
  int ii,jj;            /* generic counters */
  int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */

/* loop over _all_ cells */
  for(ii=1;ii<(n+1);ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      y_n = ii + 1;
      x_e = (jj + 1) % params.nx;
      y_s = ii - 1;
      x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);
      /* propagate densities to neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */

      tmp_ucells[ii *params.nx + jj].speeds[0]  = u[ii*params.nx + jj].speeds[0]; /* central cell, */
                                                                                     /* no movement   */
      tmp_ucells[ii *params.nx + x_e].speeds[1] = u[ii*params.nx + jj].speeds[1]; /* east */
      tmp_ucells[y_n*params.nx + jj].speeds[2]  = u[ii*params.nx + jj].speeds[2]; /* north */
      tmp_ucells[ii *params.nx + x_w].speeds[3] = u[ii*params.nx + jj].speeds[3]; /* west */
      tmp_ucells[y_s*params.nx + jj].speeds[4]  = u[ii*params.nx + jj].speeds[4]; /* south */
      tmp_ucells[y_n*params.nx + x_e].speeds[5] = u[ii*params.nx + jj].speeds[5]; /* north-east */
      tmp_ucells[y_n*params.nx + x_w].speeds[6] = u[ii*params.nx + jj].speeds[6]; /* north-west */
      tmp_ucells[y_s*params.nx + x_w].speeds[7] = u[ii*params.nx + jj].speeds[7]; /* south-west */      
      tmp_ucells[y_s*params.nx + x_e].speeds[8] = u[ii*params.nx + jj].speeds[8]; /* south-east */      
    }
  }







  return EXIT_SUCCESS;
}

int rebound(const t_param params, t_speed* u, t_speed* tmp_ucells, int* w, int n)
{
  int ii,jj;  /* generic counters */
  int size,rank;

  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  /* loop over the cells in the grid */
  for(ii=1;ii<(n+1);ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* if the cell contains an obstacle */
      if(w[ii*params.nx + jj]) {
	/* called after propagate, so taking values from scratch space
	** mirroring, and writing into main grid */
	u[ii*params.nx + jj].speeds[1] = tmp_ucells[ii*params.nx + jj].speeds[3];
	u[ii*params.nx + jj].speeds[2] = tmp_ucells[ii*params.nx + jj].speeds[4];
	u[ii*params.nx + jj].speeds[3] = tmp_ucells[ii*params.nx + jj].speeds[1];
	u[ii*params.nx + jj].speeds[4] = tmp_ucells[ii*params.nx + jj].speeds[2];
	u[ii*params.nx + jj].speeds[5] = tmp_ucells[ii*params.nx + jj].speeds[7];
	u[ii*params.nx + jj].speeds[6] = tmp_ucells[ii*params.nx + jj].speeds[8];
	u[ii*params.nx + jj].speeds[7] = tmp_ucells[ii*params.nx + jj].speeds[5];
	u[ii*params.nx + jj].speeds[8] = tmp_ucells[ii*params.nx + jj].speeds[6];
      }
    }
  }
  return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed* u, t_speed* tmp_ucells, int* w, int n)
{
  int ii,jj,kk;                 /* generic counters */
  double inv_c_sq = 3.0;
  const double w0 = 4.0/9.0;    /* weighting factor */
  const double w1 = 1.0/9.0;    /* weighting factor */
  const double w2 = 1.0/36.0;   /* weighting factor */
  double u_x,u_y;               /* av. velocities in x and y directions */
  double u2[NSPEEDS];            /* directional velocities */
  double d_equ[NSPEEDS];        /* equilibrium densities */
  double u_sq;                  /* squared velocity */
  double local_density;         /* sum of densities in a particular cell */
  double inv_local_density;
  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for(ii=1;ii<(n+1);ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* don't consider occupied cells */
      if(!w[ii*params.nx + jj]) {
  /* compute local density total */
  local_density = 0.0;
  for(kk=0;kk<NSPEEDS;kk++) {
    local_density += tmp_ucells[ii*params.nx + jj].speeds[kk];
  }
  inv_local_density = 1/local_density; 
  /* compute x velocity component */
  u_x = (tmp_ucells[ii*params.nx + jj].speeds[1] + 
         tmp_ucells[ii*params.nx + jj].speeds[5] + 
         tmp_ucells[ii*params.nx + jj].speeds[8]
         - (tmp_ucells[ii*params.nx + jj].speeds[3] + 
      tmp_ucells[ii*params.nx + jj].speeds[6] + 
      tmp_ucells[ii*params.nx + jj].speeds[7]))
    * inv_local_density;
  /* compute y velocity component */
  u_y = (tmp_ucells[ii*params.nx + jj].speeds[2] + 
         tmp_ucells[ii*params.nx + jj].speeds[5] + 
         tmp_ucells[ii*params.nx + jj].speeds[6]
         - (tmp_ucells[ii*params.nx + jj].speeds[4] + 
      tmp_ucells[ii*params.nx + jj].speeds[7] + 
      tmp_ucells[ii*params.nx + jj].speeds[8]))
    * inv_local_density;
  /* velocity squared */ 
  u_sq = u_x * u_x + u_y * u_y;
  /* directional velocity components */
  u2[1] =   u_x;        /* east */
  u2[2] =         u_y;  /* north */
  u2[3] = - u_x;        /* west */
  u2[4] =       - u_y;  /* south */
  u2[5] =   u_x + u_y;  /* north-east */
  u2[6] = - u_x + u_y;  /* north-west */
  u2[7] = - u_x - u_y;  /* south-west */
  u2[8] =   u_x - u_y;  /* south-east */
  /* equilibrium densities */
  /* zero velocity density: weight w0 */
  double wden1 = w1 * local_density;
  double wden2 = w2 * local_density;
  d_equ[0] = w0 * local_density * (1.0 - ((u_sq * inv_c_sq) / 2.0 ));
  /* axis speeds: weight w1 */
  for (kk=1;kk<5;kk++){
  d_equ[kk] = wden1 * (1.0 + (u2[kk] * inv_c_sq)
           + ((u2[kk] * u2[kk] * inv_c_sq * inv_c_sq) / (2.0) )
           - (u_sq * inv_c_sq/ 2.0));
  }
  /* diagonal speeds: weight w2 */
  for (kk=5;kk<9;kk++){
  d_equ[kk] = wden2 * (1.0 + (u2[kk] * inv_c_sq)
           + ((u2[kk] * u2[kk] * inv_c_sq * inv_c_sq) / (2.0) )
           - (u_sq * inv_c_sq/ 2.0));
  }
  /* relaxation step */
  for(kk=0;kk<NSPEEDS;kk++) {
    u[ii*params.nx + jj].speeds[kk] = (tmp_ucells[ii*params.nx + jj].speeds[kk]
             + params.omega * 
             (d_equ[kk] - tmp_ucells[ii*params.nx + jj].speeds[kk]));
  }
      }
    }
  }

  return EXIT_SUCCESS; 
}

int initialise(const char* paramfile, const char* obstaclefile,
	       t_param* params, t_speed** cells_ptr, 
	       int** obstacles_ptr, double** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE   *fp;            /* file pointer */
  int    ii,jj;          /* generic counters */
  int    xx,yy;          /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */ 
  int    retval;         /* to hold return value for checking */
  double w0,w1,w2;       /* weighting factors */

  /* open the parameter file */
  fp = fopen(paramfile,"r");
  if (fp == NULL) {
    sprintf(message,"could not open input parameter file: %s", paramfile);
    die(message,__LINE__,__FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp,"%d\n",&(params->nx));
  if(retval != 1) die ("could not read param file: nx",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->ny));
  if(retval != 1) die ("could not read param file: ny",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->maxIters));
  if(retval != 1) die ("could not read param file: maxIters",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->reynolds_dim));
  if(retval != 1) die ("could not read param file: reynolds_dim",__LINE__,__FILE__);
  retval = fscanf(fp,"%lf\n",&(params->density));
  if(retval != 1) die ("could not read param file: density",__LINE__,__FILE__);
  retval = fscanf(fp,"%lf\n",&(params->accel));
  if(retval != 1) die ("could not read param file: accel",__LINE__,__FILE__);
  retval = fscanf(fp,"%lf\n",&(params->omega));
  if(retval != 1) die ("could not read param file: omega",__LINE__,__FILE__);

  /* and close up the file */
  fclose(fp);

  /* 
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed)*(params->ny*params->nx));
  if (*cells_ptr == NULL) 
    die("cannot allocate memory for cells",__LINE__,__FILE__);
  
  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int*)*(params->ny*params->nx));
  if (*obstacles_ptr == NULL) 
    die("cannot allocate column memory for obstacles",__LINE__,__FILE__);

  /* initialise densities */
  w0 = params->density * 4.0/9.0;
  w1 = params->density      /9.0;
  w2 = params->density      /36.0;

  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      /* centre */
      (*cells_ptr)[ii*params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii*params->nx + jj].speeds[1] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[2] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[3] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii*params->nx + jj].speeds[5] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[6] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[7] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */ 
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*obstacles_ptr)[ii*params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile,"r");
  if (fp == NULL) {
    sprintf(message,"could not open input obstacles file: %s", obstaclefile);
    die(message,__LINE__,__FILE__);
  }

  /* read-in the blocked cells list */
  while( (retval = fscanf(fp,"%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* some checks */
    if ( retval != 3)
      die("expected 3 values per line in obstacle file",__LINE__,__FILE__);
    if ( xx<0 || xx>params->nx-1 )
      die("obstacle x-coord out of range",__LINE__,__FILE__);
    if ( yy<0 || yy>params->ny-1 )
      die("obstacle y-coord out of range",__LINE__,__FILE__);
    if ( blocked != 1 ) 
      die("obstacle blocked value should be 1",__LINE__,__FILE__);
    /* assign to array */
    (*obstacles_ptr)[yy*params->nx + xx] = blocked;
  }
  
  /* and close the file */
  fclose(fp);

  /* 
  ** allocate space to hold a record of the avarage velocities computed 
  ** at each timestep
  */
  *av_vels_ptr = (double*)malloc(sizeof(double)*params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, int** obstacles_ptr, double** av_vels_ptr)
{
  /* 
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

double av_velocity(const t_param params, t_speed* u, int* w, int n)
{
  int    ii,jj,kk;       /* generic counters */
  double local_density;  /* total density in cell */
  double u_x;            /* x-component of velocity for current cell */
  double u_y;            /* y-component of velocity for current cell */
  double tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0;

  /* loop over all non-blocked cells */
  for(ii=1;ii<(n+1);ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* ignore occupied cells */
      if(!w[ii*params.nx + jj]) {
	/* local density total */
	local_density = 0.0;
	for(kk=0;kk<NSPEEDS;kk++) {
	  local_density += u[ii*params.nx + jj].speeds[kk];
	}
	/* x-component of velocity */
	u_x = (u[ii*params.nx + jj].speeds[1] + 
		    u[ii*params.nx + jj].speeds[5] + 
		    u[ii*params.nx + jj].speeds[8]
		    - (u[ii*params.nx + jj].speeds[3] + 
		       u[ii*params.nx + jj].speeds[6] + 
		       u[ii*params.nx + jj].speeds[7])) / 
	  local_density;
	/* compute y velocity component */
	u_y = (u[ii*params.nx + jj].speeds[2] + 
		    u[ii*params.nx + jj].speeds[5] + 
		    u[ii*params.nx + jj].speeds[6]
		    - (u[ii*params.nx + jj].speeds[4] + 
		       u[ii*params.nx + jj].speeds[7] + 
		       u[ii*params.nx + jj].speeds[8])) /
	  local_density;
	/* accumulate the norm of x- and y- velocity components */
	tot_u += sqrt((u_x * u_x) + (u_y * u_y));
	/* increase counter of inspected cells */
      }
    }
  }

  return tot_u;
}

double calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  int    ii,jj,kk;       /* generic counters */
  int    tot_cells = 0;  /* no. of cells used in calculation */
  double local_density;  /* total density in cell */
  double u_x;            /* x-component of velocity for current cell */
  double u_y;            /* y-component of velocity for current cell */
  double tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0;

  /* loop over all non-blocked cells */
  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* ignore occupied cells */
      if(!obstacles[ii*params.nx + jj]) {
  /* local density total */
  local_density = 0.0;
  for(kk=0;kk<NSPEEDS;kk++) {
    local_density += cells[ii*params.nx + jj].speeds[kk];
  }
  /* x-component of velocity */
  u_x = (cells[ii*params.nx + jj].speeds[1] + 
        cells[ii*params.nx + jj].speeds[5] + 
        cells[ii*params.nx + jj].speeds[8]
        - (cells[ii*params.nx + jj].speeds[3] + 
           cells[ii*params.nx + jj].speeds[6] + 
           cells[ii*params.nx + jj].speeds[7])) / 
    local_density;
  /* compute y velocity component */
  u_y = (cells[ii*params.nx + jj].speeds[2] + 
        cells[ii*params.nx + jj].speeds[5] + 
        cells[ii*params.nx + jj].speeds[6]
        - (cells[ii*params.nx + jj].speeds[4] + 
           cells[ii*params.nx + jj].speeds[7] + 
           cells[ii*params.nx + jj].speeds[8])) /
    local_density;
  /* accumulate the norm of x- and y- velocity components */
  tot_u += sqrt((u_x * u_x) + (u_y * u_y));
  /* increase counter of inspected cells */
  ++tot_cells;
      }
    }
  }
  
  return (tot_u / (double)tot_cells) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells)
{
  int ii,jj,kk;        /* generic counters */
  double total = 0.0;  /* accumulator */

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      for(kk=0;kk<NSPEEDS;kk++) {
	total += cells[ii*params.nx + jj].speeds[kk];
      }
    }
  }
  
  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels)
{
  FILE* fp;                     /* file pointer */
  int ii,jj,kk;                 /* generic counters */
  const double c_sq = 1.0/3.0;  /* sq. of speed of sound */
  double local_density;         /* per grid cell sum of densities */
  double pressure;              /* fluid pressure in grid cell */
  double u_x;                   /* x-component of velocity in grid cell */
  double u_y;                   /* y-component of velocity in grid cell */
  double u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE,"w");
  if (fp == NULL) {
    die("could not open file output file",__LINE__,__FILE__);
  }

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* an occupied cell */
      if(obstacles[ii*params.nx + jj]) {
	u_x = u_y = u = 0.0;
	pressure = params.density * c_sq;
      }
      /* no obstacle */
      else {
	local_density = 0.0;
	for(kk=0;kk<NSPEEDS;kk++) {
	  local_density += cells[ii*params.nx + jj].speeds[kk];
	}
	/* compute x velocity component */
	u_x = (cells[ii*params.nx + jj].speeds[1] + 
	       cells[ii*params.nx + jj].speeds[5] +
	       cells[ii*params.nx + jj].speeds[8]
	       - (cells[ii*params.nx + jj].speeds[3] + 
		  cells[ii*params.nx + jj].speeds[6] + 
		  cells[ii*params.nx + jj].speeds[7]))
	  / local_density;
	/* compute y velocity component */
	u_y = (cells[ii*params.nx + jj].speeds[2] + 
	       cells[ii*params.nx + jj].speeds[5] + 
	       cells[ii*params.nx + jj].speeds[6]
	       - (cells[ii*params.nx + jj].speeds[4] + 
		  cells[ii*params.nx + jj].speeds[7] + 
		  cells[ii*params.nx + jj].speeds[8]))
	  / local_density;
	/* compute norm of velocity */
	u = sqrt((u_x * u_x) + (u_y * u_y));
	/* compute pressure */
	pressure = local_density * c_sq;
      }
      /* write to file */
      fprintf(fp,"%d %d %.12E %.12E %.12E %.12E %d\n",jj,ii,u_x,u_y,u,pressure,obstacles[ii*params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE,"w");
  if (fp == NULL) {
    die("could not open file output file",__LINE__,__FILE__);
  }
  for (ii=0;ii<params.maxIters;ii++) {
    fprintf(fp,"%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n",message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
