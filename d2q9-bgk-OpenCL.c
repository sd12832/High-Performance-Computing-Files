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
#include "err_code.h"
#include <unistd.h>

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
// typedef struct {
//   float speeds[NSPEEDS];
// } t_speed;

enum boolean { FALSE, TRUE };

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
	       t_param* params, float** cells_ptr, float** tmp_cells_ptr, 
	       int** obstacles_ptr, float** av_vels_ptr);

/* 
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int write_values(const t_param params, float* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
	     int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, float* cells);

/* Kernel Source Extractor Function */
char * getKernelSource(char *filename);

/* compute average velocity */
double av_velocity(const t_param params, float* cells, int* obstacles);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, float* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char *file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/

//--------------------------------------------------------------------------------
// Kernel Source Extractor
//--------------------------------------------------------------------------------

char * getKernelSource(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error: Could not open kernel source file\n");
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    int len = ftell(file) + 1;
    rewind(file);

    char *source = (char *)calloc(sizeof(char), len);
    if (!source)
    {
        fprintf(stderr, "Error: Could not allocate memory for source string\n");
        exit(EXIT_FAILURE);
    }
    fread(source, sizeof(char), len, file);
    fclose(file);
    return source;
    printf("Inside Kernel Source");
}

//--------------------------------------------------------------------------------
// Context and Device Initializer
//--------------------------------------------------------------------------------

void init_context_bcp3(cl_context* context, cl_device_id* device)
{
    
    // Get the index into the device array allocated by the queue on BCP3
    

    // --------------------------------------------------------------------------------
    // GPU File
    // --------------------------------------------------------------------------------

    char* path = getenv("PBS_GPUFILE");
    if (path == NULL)
    {
        fprintf(stderr, "Error: PBS_GPUFILE environment variable not set by queue\n");
        exit(-1);
    }
    FILE* gpufile = fopen(path, "r");
    if (gpufile == NULL)
    {
        fprintf(stderr, "Error: PBS_GPUFILE not found\n");
        exit(-1);
    }

    size_t max_length = 100;
    char* line = (char *)malloc(sizeof(char)*max_length);

    size_t len = getline(&line, &max_length, gpufile);

    int device_index = (int) strtol(&line[len-2], NULL, 10);

    free(line);
    fclose(gpufile);

    cl_int err;

    // Query platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) {fprintf(stderr, "Error counting platforms: %d\n", err); exit(-1);}

    cl_platform_id platforms[num_platforms];
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {fprintf(stderr, "Error getting platforms: %d\n", err); exit(-1);}

    // Get devices for each platform - stop when found a GPU
    cl_uint max_devices = 8;
    cl_device_id devices[max_devices];
    cl_uint num_devices;

    int i;
    for (i = 0; i < num_platforms; i++)
    {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, max_devices, devices, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0)
            break;
    }
    if (num_devices == 0)
    {
        fprintf(stderr, "Error: no GPUs found");
        exit(-1);
    }

    // Assign the device id
    *device = devices[device_index];

    // Create the context
    const cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0};
    *context = clCreateContext(properties, 1, device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {fprintf(stderr, "Error creating context: %d\n", err); exit(-1);}

    // --------------------------------------------------------------------------------
    // CPU File
    // --------------------------------------------------------------------------------

    // cl_int err;

    // // Query platforms
    // cl_uint num_platforms;
    // err = clGetPlatformIDs(0, NULL, &num_platforms);
    // if (err != CL_SUCCESS) {fprintf(stderr, "Error counting platforms: %d\n", err); exit(-1);}

    // cl_platform_id platforms[num_platforms];
    // err = clGetPlatformIDs(num_platforms, platforms, NULL);
    // if (err != CL_SUCCESS) {fprintf(stderr, "Error getting platforms: %d\n", err); exit(-1);}

    // // Get devices for each platform - stop when found a GPU
    // cl_uint max_devices = 8;
    // cl_device_id devices[max_devices];
    // cl_uint num_devices;

    // int i;
    // for (i = 0; i < num_platforms; i++)
    // {
    //     err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, max_devices, devices, &num_devices);
    //     if (err == CL_SUCCESS && num_devices > 0)
    //         break;
    // }
    // if (num_devices == 0)
    // {
    //     fprintf(stderr, "Error: no GPUs found");
    //     exit(-1);
    // }

    // // Assign the device id
    // *device = devices[0];

    // // Create the context
    // const cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0};
    // *context = clCreateContext(properties, 1, device, NULL, NULL, &err);
    // if (err != CL_SUCCESS) {fprintf(stderr, "Error creating context: %d\n", err); exit(-1);}    

}

//--------------------------------------------------------------------------------
// Main Function
//--------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  float* cells     = NULL;    /* grid containing fluid densities */
  float* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float*  av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */
  int      ii, jj;                  /* generic counter */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic,toc;               /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */
  int err; 
  cl_mem d_cells;                     // device memory used for the input  a vector
  cl_mem d_tmp_cells;                     //
  cl_mem d_obstacles;   
  cl_mem d_av_vels;                     // device memory used for the output c vector
  cl_mem global_sums;  
  cl_mem d_iteration_cntr;              

  /* parse the command line */
  if(argc != 3) { 
    usage(argv[0]);
  }
  else{
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);


//--------------------------------------------------------------------------------
// Testing for obtaining Files
//--------------------------------------------------------------------------------  
//--------------------------------------------------------------------------------  
//--------------------------------------------------------------------------------  
//--------------------------------------------------------------------------------  
//--------------------------------------------------------------------------------  
//--------------------------------------------------------------------------------  

   // printf("PREValue for cells: %lf\n", cells[params.nx+4].speeds[1]);
  // printf("PREvalue for obstacles: %d\n", obstacles[params.nx+6]);

//--------------------------------------------------------------------------------
// Creating variables of sizes
//--------------------------------------------------------------------------------

  size_t global = {params.nx};
  size_t Tglobal[] = {params.nx, params.ny};
  // const size_t local = {32}; 
  // const size_t Tlocal[] = {32, 1}; 
  size_t local = {16}; 
  size_t Tlocal[] = {16,16}; 

if (params.nx == 700 && params.ny == 500) {

  global = params.nx;
  Tglobal[0] = params.nx;
  Tglobal[1] = params.ny;
  local = 10;
  Tlocal[0] = 10; 
  Tlocal[1] = 10; 

}

//--------------------------------------------------------------------------------
// Creating a context, queue and device and kernels
//--------------------------------------------------------------------------------

  cl_context context;
  cl_device_id device;
  cl_command_queue commands;      // compute command queue

  init_context_bcp3(&context, &device);

  commands = clCreateCommandQueue(context, device, 0, &err);
  checkError(err, "Creating command queue");

  char options[] = "-cl-single-precision-constant -cl-no-signed-zeros -cl-mad-enable";
  // "/-cl-single-precision-constant -cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable"

// Kernel accelerate_flow //////////////////////////////////////////////////////////////////////////////////

  cl_program       program_af;       // compute program
  cl_kernel        a_f;       // compute kernel
  char * KernelSource_af;    // kernel source string

  KernelSource_af = getKernelSource("lb_af.cl");

  program_af = clCreateProgramWithSource(context, 1, (const char **) & KernelSource_af, NULL, &err);
  checkError(err, "Creating program");
  // Build the program
  err = clBuildProgram(program_af, 0, NULL, options, NULL, NULL);

  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];
    printf("Error: Failed to build program executable!\n%s\n", err_code(err));
    clGetProgramBuildInfo(program_af, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    return EXIT_FAILURE;
  }

  // Create the compute kernel from the program
  a_f = clCreateKernel(program_af, "accelerate_flow", &err);
  checkError(err, "Creating kernel a_f");

// Kernel propagate /////////////////////////////////////////////////////////////////////////////////////////

  cl_program       program_p;       // compute program
  cl_kernel        prop;       // compute kernel
  char * KernelSource_p;    // kernel source string

  KernelSource_p = getKernelSource("lb_p.cl");

  program_p = clCreateProgramWithSource(context, 1, (const char **) & KernelSource_p, NULL, &err);  
  checkError(err, "Creating program");
  // Build the program
  err = clBuildProgram(program_p, 0, NULL, options, NULL, NULL);

  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[20480];
    printf("Error: Failed to build program executable!\n%s\n", err_code(err));
    clGetProgramBuildInfo(program_p, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    return EXIT_FAILURE;
  }

  prop = clCreateKernel(program_p, "propagate", &err);
  // Create the compute kernel from the program
  checkError(err, "Creating kernel prop");

// Kernel av_velocity //////////////////////////////////////////////////////////////////////////////////////

  cl_program       program_av;       // compute program
  cl_kernel        av;       // compute kernel
  char * KernelSource_av;    // kernel source string

  KernelSource_av = getKernelSource("lb_av.cl");

  program_av = clCreateProgramWithSource(context, 1, (const char **) & KernelSource_av, NULL, &err);
  checkError(err, "Creating program");
  // Build the program
  err = clBuildProgram(program_av, 0, NULL, options, NULL, NULL);

  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];
    printf("Error: Failed to build program executable!\n%s\n", err_code(err));
    clGetProgramBuildInfo(program_av, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    return EXIT_FAILURE;
  }

  // Create the compute kernel from the program
  av = clCreateKernel(program_av, "av_velocity", &err);
  checkError(err, "Creating kernel av");


//--------------------------------------------------------------------------------
// Creating the buffers
//--------------------------------------------------------------------------------

  d_cells  = clCreateBuffer(context,  CL_MEM_READ_WRITE,  (sizeof(float) * NSPEEDS * params.ny * params.nx) , NULL, &err);
  checkError(err, "Creating buffer cells");

  d_tmp_cells  = clCreateBuffer(context,  CL_MEM_READ_WRITE,  (sizeof(float) * NSPEEDS * params.ny * params.nx) , NULL, &err);
  checkError(err, "Creating buffer tmp_cells");

  d_obstacles  = clCreateBuffer(context,  CL_MEM_READ_WRITE,  (sizeof(int) * params.ny * params.nx) , NULL, &err);
  checkError(err, "Creating buffer obstacles");

  d_av_vels = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(float)*params.maxIters, NULL, &err);
  checkError(err, "Creating buffer av_vels");

  global_sums = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(float) * ((params.ny * params.nx)/(Tlocal[0]*Tlocal[1])), NULL, &err);
  checkError(err, "Creating buffer global_sums");

  d_iteration_cntr = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
  checkError(err, "Creating buffer global_sums");

  // for (ii=0;ii<params.maxIters;ii++) {

  //--------------------------------------------------------------------------------
  // Writing to the buffers
  //--------------------------------------------------------------------------------
  // printf("Value- for cells: %lf\n", cells[params.nx+4].speeds[1]);
  // printf("value- for obstacles: %d\n", obstacles[params.nx+6]);

  int h_iteration_cntr = 0;

  err = clEnqueueWriteBuffer(commands, d_iteration_cntr, CL_TRUE, 0, sizeof(int), &h_iteration_cntr, 0, NULL, NULL);
  checkError(err, "Copying cells to device at d_cells");

  err = clEnqueueWriteBuffer(commands, d_cells, CL_TRUE, 0, (sizeof(float) * NSPEEDS * params.ny * params.nx), cells, 0, NULL, NULL);
  checkError(err, "Copying cells to device at d_cells");

  err = clEnqueueWriteBuffer(commands, d_obstacles, CL_TRUE, 0, (sizeof(int) * params.ny * params.nx), obstacles, 0, NULL, NULL);
  checkError(err, "Copying obstacles to device at d_obstacles");
  clFinish(commands);

  //--------------------------------------------------------------------------------
  // Setting Arguments
  //--------------------------------------------------------------------------------

  // Kernel accelerate_flow //////////////////////////////////////////////////////////////////////////////

  err = clSetKernelArg(a_f, 0, sizeof(t_param), &params);
  err |= clSetKernelArg(a_f, 2, sizeof(cl_mem), &d_obstacles);
  checkError(err, "Setting kernel arguments for af");

  // printf("Three17a\n"); 
 
  // Kernel propogate /////////////////////////////////////////////////////////////////////////////////////

  err = clSetKernelArg(prop, 0, sizeof(t_param), &params);
  err |= clSetKernelArg(prop, 3, sizeof(cl_mem), &d_obstacles);
  err |= clSetKernelArg(prop, 4, sizeof(float)*Tlocal[0]*Tlocal[1], NULL);
  err |= clSetKernelArg(prop, 5, sizeof(cl_mem), &global_sums);
  checkError(err, "Setting kernel arguments for p");
 
  // Kernel av_velocity //////////////////////////////////////////////////////////////////////////////

  int N = 0; 
  err = clSetKernelArg(av, 0, sizeof(cl_mem), &global_sums);
  err |= clSetKernelArg(av, 1, sizeof(cl_mem), &d_iteration_cntr);
  err |= clSetKernelArg(av, 2, sizeof(cl_mem), &d_av_vels);
  checkError(err, "Setting kernel arguments for av");

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr,NULL);
  tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);



  int tot_cells; 

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* ignore occupied cells */
      if(!obstacles[ii*params.nx + jj]) {
        tot_cells++; 
      }
    }
  }

  //--------------------------------------------------------------------------------
  // Running the Iterations and Enqueing the Kernels 
  //--------------------------------------------------------------------------------

//   //////////////////////////////////////////////////////
//  /**/               printf("Threea2\n");           /**/
// //////////////////////////////////////////////////////

  for (ii=0;ii<params.maxIters;ii++) {
  // for (ii=0;ii<1;ii++) {

  if (ii % 2 == 0 ) {
    err |= clSetKernelArg(a_f, 1, sizeof(cl_mem), &d_cells);

    err |= clSetKernelArg(prop, 1, sizeof(cl_mem), &d_cells);
    err |= clSetKernelArg(prop, 2, sizeof(cl_mem), &d_tmp_cells);
    checkError(err, "Setting kernel arguments");

  } else {
    err |= clSetKernelArg(a_f, 1, sizeof(cl_mem), &d_tmp_cells);

    err |= clSetKernelArg(prop, 1, sizeof(cl_mem), &d_tmp_cells);
    err |= clSetKernelArg(prop, 2, sizeof(cl_mem), &d_cells);
      checkError(err, "Setting kernel arguments");
  }

  // for (ii=0;ii<1;ii++) {

  // double tic1;

  // gettimeofday(&timstr,NULL);
  // tic1=timstr.tv_sec+(timstr.tv_usec/1000000.0);

  // Kernel accelerate_flow ///////////////////////////////////////////////////////////////////////////

  err = clEnqueueNDRangeKernel(commands, a_f, 1, NULL, &global, &local, 0, NULL, NULL);
  checkError(err, "Enqueueing kernel a_f");
  //clWaitForEvents(1, &event1);

  // double tic2;
  // gettimeofday(&timstr,NULL);
  // tic2=timstr.tv_sec+(timstr.tv_usec/1000000.0);

  // printf("Elapsed time for accelerate_flow:\t\t\t%.6lf (s)\n", tic2-tic1);

  // Kernel propogate /////////////////////////////////////////////////////////////////////////////////

  err = clEnqueueNDRangeKernel(commands, prop, 2, NULL, Tglobal, Tlocal, 0, NULL, NULL);
  checkError(err, "Enqueueing kernel prop");

  // double tic3;
  // gettimeofday(&timstr,NULL);
  // tic3=timstr.tv_sec+(timstr.tv_usec/1000000.0);

  // printf("Elapsed time for prop:\t\t\t%.6lf (s)\n", tic3-tic2);

  // Kernel av_velocity //////////////////////////////////////////////////////////////////////////////


  err = clEnqueueNDRangeKernel(commands, av, 2, NULL, Tglobal, Tlocal, 0, NULL, NULL);
  checkError(err, "Enqueueing kernel av");

  // double tic4;
  // gettimeofday(&timstr,NULL);
  // tic4=timstr.tv_sec+(timstr.tv_usec/1000000.0);

  // printf("Elapsed time for prop:\t\t\t%.6lf (s)\n", tic4-tic3);

  }

    //--------------------------------------------------------------------------------
  // Writing back all the values from the buffers
  //--------------------------------------------------------------------------------

  // cl_event event6;
  err = clEnqueueReadBuffer(commands, d_cells, CL_TRUE, 0, sizeof(float) * NSPEEDS * params.ny * params.nx, cells, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array back to cells!\n%s\n", err_code(err));
    exit(1);
  }
  // sizeof(t_speed) * params.nx* params.ny
  // printf("Value+ for cells: %lf\n", cells[params.nx+4].speeds[1]);

  // cl_event event7;
  err = clEnqueueReadBuffer(commands, d_av_vels, CL_TRUE, 0, sizeof(float)*params.maxIters, av_vels, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array back to av_vels!\n%s\n", err_code(err));
    exit(1);
  }

  // cl_event events2[] = {event6, event7}; 

  //clWaitForEvents(2, events2);

  //--------------------------------------------------------------------------------
  //Calculating av_vels
  //--------------------------------------------------------------------------------

  /*int tot_cells; 

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* ignore occupied cells */
     /* if(!obstacles[ii*params.nx + jj]) {
        tot_cells++; 
      }
    }
  }*/

  for(ii=0;ii<params.maxIters;ii++) {
    // printf("%d =", av_vels[ii]);
    av_vels[ii] = av_vels[ii]/(float)tot_cells; 
    // printf("%d\n", av_vels[ii]);
  }

//--------------------------------------------------------------------------------
// Testing for write-back of Files
//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------  
//--------------------------------------------------------------------------------  
//--------------------------------------------------------------------------------  
//--------------------------------------------------------------------------------  
//--------------------------------------------------------------------------------  
//--------------------------------------------------------------------------------    

  // printf("Value for cells: %lf\n", cells[params.nx+4].speeds[1]);
  // printf("value for obstacles: %d\n", obstacles[params.nx+6]);

  // sizeof(double)* params.maxIters

  gettimeofday(&timstr,NULL);
  toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr=ru.ru_utime;        
  usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
  timstr=ru.ru_stime;        
  systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);

  /* write final values and free memory */
  printf("\n==done==\n");
  printf("Reynolds number:\t\t%.12E\n",calc_reynolds(params,cells,obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc-tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params,cells,obstacles,av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  //--------------------------------------------------------------------------------
  // Releasing the Memory Objects
  //--------------------------------------------------------------------------------
  clReleaseMemObject(d_cells);
  clReleaseMemObject(d_tmp_cells);
  clReleaseMemObject(d_obstacles);
  clReleaseMemObject(d_av_vels);
  clReleaseProgram(program_af);
  clReleaseKernel(a_f);
  clReleaseProgram(program_p);
  clReleaseKernel(prop);
  // clReleaseProgram(program_r);
  // clReleaseKernel(reb);
  // clReleaseProgram(program_c);
  // clReleaseKernel(coll);
  clReleaseProgram(program_av);
  clReleaseKernel(av);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
  return EXIT_SUCCESS;
}

int initialise(const char* paramfile, const char* obstaclefile,
	       t_param* params, float** cells_ptr, float** tmp_cells_ptr, 
	       int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE   *fp;            /* file pointer */
  int    ii,jj;          /* generic counters */
  int    xx,yy;          /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */ 
  int    retval;         /* to hold return value for checking */
  float w0,w1,w2;       /* weighting factors */

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

  *cells_ptr = (float*)malloc(sizeof(float)*NSPEEDS*(params->ny*params->nx));
  if (*cells_ptr == NULL) 
    die("cannot allocate memory for cells",__LINE__,__FILE__);

   //'helper' grid, used as scratch space 
  *tmp_cells_ptr = (float*)malloc(sizeof(float)*NSPEEDS*(params->ny*params->nx));
  if (*tmp_cells_ptr == NULL) 
    die("cannot allocate memory for tmp_cells",__LINE__,__FILE__);
  
  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int*)*(params->ny*params->nx));
  if (*obstacles_ptr == NULL) 
    die("cannot allocate column memory for obstacles",__LINE__,__FILE__);


  /* initialise densities */
  w0 = params->density * 4.0/9.0;
  w1 = params->density      /9.0;
  w2 = params->density      /36.0;

  //speed 0
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*cells_ptr)[ii*params->nx + jj] = w0;
    }
  }

  int gridsize = params->ny * params->ny; 

  //speed 1
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*cells_ptr)[gridsize + (ii*params->nx + jj)] = w1;
    }
  }

  //speed 2
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*cells_ptr)[(gridsize*2) + (ii*params->nx + jj)] = w1;
    }
  }

  //speed 3
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*cells_ptr)[(gridsize*3) + (ii*params->nx + jj)] = w1;
    }
  }

  //speed 4
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*cells_ptr)[(gridsize*4) + (ii*params->nx + jj)] = w1;
    }
  }

  //speed 5
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*cells_ptr)[(gridsize*5) + (ii*params->nx + jj)] = w2;
    }
  }

  //speed 6
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*cells_ptr)[(gridsize*6) + (ii*params->nx + jj)] = w2;
    }
  }

  //speed 7
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*cells_ptr)[(gridsize*7) + (ii*params->nx + jj)] = w2;
    }
  }

  //speed 8
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*cells_ptr)[(gridsize*8) + (ii*params->nx + jj)] = w2;
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
  *av_vels_ptr = (float*)malloc(sizeof(float)*params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
	     int** obstacles_ptr, float** av_vels_ptr)
{
  /* 
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

double av_velocity(const t_param params, float* cells, int* obstacles)
{
  int    ii,jj,kk;       /* generic counters */
  int    tot_cells = 0;  /* no. of cells used in calculation */
  double local_density;  /* total density in cell */
  double u_x;            /* x-component of velocity for current cell */
  double u_y;            /* y-component of velocity for current cell */
  double tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0;
  int sn; 
  int gridsize = params.ny * params.ny;

  /* loop over all non-blocked cells */
  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* ignore occupied cells */
      if(!obstacles[ii*params.nx + jj]) {
  /* local density total */
  local_density = 0.0;
  for(kk=0;kk<NSPEEDS;kk++) {
    local_density += cells[(kk*gridsize) + (ii*params.nx + jj)];
  }
  /* x-component of velocity */
  u_x = (cells[(gridsize) + (ii*params.nx + jj)] + 
        cells[(5*gridsize) + (ii*params.nx + jj)] + 
        cells[(8*gridsize) + (ii*params.nx + jj)]
        - (cells[(3*gridsize) + (ii*params.nx + jj)] + 
           cells[(6*gridsize) + (ii*params.nx + jj)] + 
           cells[(7*gridsize) + (ii*params.nx + jj)])) / 
    local_density;
  /* compute y velocity component */
  u_y = (cells[(2*gridsize) + (ii*params.nx + jj)] + 
        cells[(5*gridsize) + (ii*params.nx + jj)] + 
        cells[(6*gridsize) + (ii*params.nx + jj)]
        - (cells[(4*gridsize) + (ii*params.nx + jj)] + 
           cells[(7*gridsize) + (ii*params.nx + jj)] + 
           cells[(8*gridsize) + (ii*params.nx + jj)])) /
    local_density;
  /* accumulate the norm of x- and y- velocity components */
  tot_u += sqrt((u_x * u_x) + (u_y * u_y));
  /* increase counter of inspected cells */
  ++tot_cells;
      }
    }
  }

  return tot_u / (double)tot_cells;
}

double calc_reynolds(const t_param params, float* cells, int* obstacles)
{
  const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  
  return av_velocity(params,cells,obstacles) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, float* cells)
{
  int ii,jj,kk;        /* generic counters */
  double total = 0.0;  /* accumulator */

  int gridsize = params.ny * params.ny;

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      for(kk=0;kk<NSPEEDS;kk++) {
	       total += cells[(kk*gridsize) + (ii*params.nx + jj)];
      }
    }
  }
  
  return total;
}

int write_values(const t_param params, float* cells, int* obstacles, float* av_vels)
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

  int gridsize = params.ny * params.ny;

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
      	  local_density += cells[(kk*gridsize) + (ii*params.nx + jj)];
    	}
	/* compute x velocity component */
  u_x = (cells[(gridsize) + (ii*params.nx + jj)] + 
        cells[(5*gridsize) + (ii*params.nx + jj)] + 
        cells[(8*gridsize) + (ii*params.nx + jj)]
        - (cells[(3*gridsize) + (ii*params.nx + jj)] + 
           cells[(6*gridsize) + (ii*params.nx + jj)] + 
           cells[(7*gridsize) + (ii*params.nx + jj)])) / 
    local_density;
  /* compute y velocity component */
  u_y = (cells[(2*gridsize) + (ii*params.nx + jj)] + 
        cells[(5*gridsize) + (ii*params.nx + jj)] + 
        cells[(6*gridsize) + (ii*params.nx + jj)]
        - (cells[(4*gridsize) + (ii*params.nx + jj)] + 
           cells[(7*gridsize) + (ii*params.nx + jj)] + 
           cells[(8*gridsize) + (ii*params.nx + jj)])) /
    local_density;
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
