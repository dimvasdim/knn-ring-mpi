/**
    Author: Dimitriadis Vasileios 8404
    Faculty of Electrical and Computer Engineering AUTH
    2nd assignment at Parallel and Distributed Systems (7th semester)
    This is a parallel implementation of knn algorithm using mpi.
    In this version the data are splitted in equal parts to each process(blocks).
    Then every process is responsible for sending data to the next process
    while receiving from the previous one (ring network).
    Non-blocking communications are being used.
  **/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define dimensions 30
#define points 60000
#define MAXD 10000
#define NUML 10

struct timeval startwtime, endwtime;
double seq_time;

void fill_obs(double **obs, char *filename);
void fill_labels(int *labels, char *filename);
void init_dis(double **a);
void self_knn_search(double **obs, double **dis, int **ind);
double self_find_distance(int i, int j, double **obs);
void self_check(int i, int j, double distance, double **dis, int **ind);
void knn_search(double **new_obs, double **obs, double **dis, int **ind, int count);
double find_distance(int i, int j, double **obs, double **new_obs);
void check(int i, int j, double distance, double **dis, int **ind, int count);
void self_find_labels(int **lab, int *labels, int **ind);
void find_labels(int **lab, int *new_labels, int **ind, int count);
int matching(int **lab, int *labels);
int most_common(int i, int **lab);
void show_labels(int **lab);
void show_results(double **dis, int **ind);
void free_d(double **a);
void free_int(int **a);
int **alloc_2d_int(int rows, int cols);
double **alloc_2d_d(int rows, int cols);
void init_neighbors();
void swap_tables(double **temp, double **new_obs);
void swap_arrays(int *temp_labels, int *new_labels);
int testing(int **ind, char *filename);

int k; //Need k in a lot of functions so it's better to be global
int chunk; //The 'portion' of the points that every process will handle.
int rank, numtasks, prev, next, tag = 0;

MPI_Status stats[2];
MPI_Request req[2];

int main(int argc, char **argv)
{
  if (argc != 5)
  {
    printf("Need as arguments:\n");
    printf("k, which is the number of neighbors\n");
    printf(",the txt file with the observations\n");
    printf(",the txt file with the labels\n");
    printf("and the txt files with the indexes for validation...");
    exit(1);
  }

  int i, c = 0;
  int local_sum, total_sum = 0;
  float matches;

  c = MPI_Init(&argc, &argv);
  if (c != MPI_SUCCESS)
  {
    printf("Error at mpi init\n");
    MPI_Finalize();
    exit(1);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  k = atoi(argv[1]);

  if ((points % numtasks) != 0)
  {
    printf("You should pick another number of processes, so as to divide ");
    printf("the number of points without a reminder...\n");
    MPI_Finalize();
    exit(1);
  }
  chunk = points / numtasks;

  //Creation of the table with the observations.
  double **obs = alloc_2d_d(chunk, dimensions);

  fill_obs(obs, argv[2]);

  /** Creation of the table with the observations
      that are being received from previous process
      and are being sent to the next one.
  **/
  double **new_obs = alloc_2d_d(chunk, dimensions);

  //Creation of a temporary table for receiving data.
  double **temp = alloc_2d_d(chunk, dimensions);

  //Creation of the table for the distances between points.
  double **dis = alloc_2d_d(chunk, k);

  init_dis(dis);

  //Creation of the table for the indexes of the points.
  int **ind = alloc_2d_int(chunk, k);

  init_neighbors();

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
  {
    gettimeofday (&startwtime, NULL);
  }
  swap_tables(obs, new_obs);
  MPI_Irecv(&(temp[0][0]), chunk*dimensions, MPI_DOUBLE, prev, tag, MPI_COMM_WORLD, &req[0]);
  MPI_Isend(&(new_obs[0][0]), chunk*dimensions, MPI_DOUBLE, next, tag, MPI_COMM_WORLD, &req[1]);
  self_knn_search(obs, dis, ind);
  MPI_Waitall(2, req, stats);
  for (i=0; i<(numtasks-1); i++)
  {
    swap_tables(temp, new_obs);
    if (i < (numtasks-2))
    {
      MPI_Irecv(&(temp[0][0]), chunk*dimensions, MPI_DOUBLE, prev, tag, MPI_COMM_WORLD, &req[0]);
      MPI_Isend(&(new_obs[0][0]), chunk*dimensions, MPI_DOUBLE, next, tag, MPI_COMM_WORLD, &req[1]);
    }
    knn_search(new_obs, obs, dis, ind, i);
    if (i < (numtasks-2))
    {
      MPI_Waitall(2, req, stats);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
  {
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
                          + endwtime.tv_sec - startwtime.tv_sec);
    printf("Time needed for knn search is %f sec\n", seq_time);
    //show_results(dis, ind);
  }

  free_d(obs); //Don't need the observations anymore.
  free_d(new_obs);
  free_d(temp);

  /** Create a table with (chunk)# of rows and (k)# of columns.
      The k columns will be the labels that match to the indexes.
  **/
  int **lab = alloc_2d_int(chunk, k);

  //Creation of the array of the labels.
  int *labels = (int *)malloc(chunk * sizeof(int));

  fill_labels(labels, argv[3]);

  /** Creation of the array with the labels that
      are being received from previous process
      and are being sent to the next one.
  **/
  int *new_labels = (int *)malloc(chunk * sizeof(int));

  //Creation of a temporary array for the labels received.
  int *temp_labels = (int *)malloc(chunk * sizeof(int));

  if (rank == 0)
  {
    gettimeofday (&startwtime, NULL);
  }
  swap_arrays(labels, new_labels);
  MPI_Irecv(&(temp_labels[0]), chunk, MPI_INT, prev, tag, MPI_COMM_WORLD, &req[0]);
  MPI_Isend(&(new_labels[0]), chunk, MPI_INT, next, tag, MPI_COMM_WORLD, &req[1]);
  self_find_labels(lab, labels, ind);
  MPI_Waitall(2, req, stats);
  for (i=0; i<(numtasks-1); i++)
  {
    swap_arrays(temp_labels, new_labels);
    if (i < (numtasks-2))
    {
      MPI_Irecv(&(temp_labels[0]), chunk, MPI_INT, prev, tag, MPI_COMM_WORLD, &req[0]);
      MPI_Isend(&(new_labels[0]), chunk, MPI_INT, next, tag, MPI_COMM_WORLD, &req[1]);
    }
    find_labels(lab, new_labels, ind, i);
    if (i < (numtasks-2))
    {
      MPI_Waitall(2, req, stats);
    }
  }
  local_sum = matching(lab, labels);
  MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  //show_labels(lab);
  if (rank == 0)
  {
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
                          + endwtime.tv_sec - startwtime.tv_sec);
    printf("Time needed for labels is %f sec\n", seq_time);
    //show_labels(lab);
    matches = (float)total_sum / (float)points *100;
    printf("Total matches are %f%%\n", matches);
  }
  local_sum = testing(ind, argv[4]);
  MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0)
  {
    printf("Total errors in indexes: %d\n", total_sum);
    printf("\n \n \n");
  }

  free_int(lab);
  free(labels);
  free(new_labels);
  free(temp_labels);

  free_d(dis);
  free_int(ind);
  MPI_Finalize();
  return (0);
}

//Neighbor processes are subsequent to each other.
void init_neighbors()
{
  prev = rank - 1;
  next = rank + 1;
  if (rank == 0)
  {
    prev = numtasks - 1;
  }
  if (rank == (numtasks - 1))
  {
    next = 0;
  }
}

//Function to fill the table with the observations.
void fill_obs(double **obs, char *filename)
{
  FILE *fin;
  /** The txt file contains all the observations in decimal numbers.
      Each line refers to a specific point and the dimensions of each
      point are splitted by a tab.
  **/

  char *str = (char *)malloc(2 * dimensions * sizeof(double));
  char *token = (char *)malloc(sizeof(double));
  fin = fopen(filename, "r");
  if (fin == NULL)
  {
    printf("Error opening the file...");
    exit(1);
  }
  int i = 0;
  int j;
  //Skip lines of the files if needed to store the right data.
  for (j=0; j<(rank * chunk); j++)
  {
    str = fgets(str, 2 * dimensions * sizeof(double), fin);
  }
  str = fgets(str, 2 * dimensions * sizeof(double), fin); //get a line of the txt file, which refers to one point.
  while (str != NULL && i < chunk)
  {
    token = strtok(str, "\t"); //get one dimension per recursion.
    j = 0;
    while (token != NULL && j < dimensions)
    {
      obs[i][j] = atof(token);
      token = strtok(NULL, "\t");
      j++;
    }
    str = fgets(str, 2 * dimensions * sizeof(double), fin);
    i++;
  }
  fclose(fin);
  free(str);
  free(token);
}

//Function to fill the array of the labels.
void fill_labels(int *labels, char *filename)
{
  FILE *fin;
  char *str = (char *)malloc(sizeof(int)+1);
  fin = fopen(filename, "r");
  if (fin == NULL)
  {
    printf("Error opening the file...");
    exit(1);
  }
  int i;
  //Skip lines of the files if needed to store the right data.
  for (i=0; i<(rank * chunk); i++)
  {
    str = fgets(str, sizeof(int)+1, fin);
  }
  str = fgets(str, sizeof(int)+1, fin);
  i = 0;
  while (str != NULL && i < chunk)
  {
    labels[i] = atoi(str);
    str = fgets(str, sizeof(int)+1, fin);
    i++;
  }
  fclose(fin);
  free(str);
}

//Replacing table new_ops with the values of table temp.
void swap_tables(double **temp, double **new_obs)
{
  int i,j;
  for (i=0; i<chunk; i++)
  {
    for (j=0; j<dimensions; j++)
    {
      new_obs[i][j] = temp[i][j];
    }
  }
}

void swap_arrays(int *temp_labels, int *new_labels)
{
  for (int i=0; i<chunk; i++)
  {
    new_labels[i] = temp_labels[i];
  }
}

void knn_search(double **new_obs, double **obs, double **dis, int **ind, int count)
{
  int i,j;
  double distance;
  for (i=0; i<chunk; i++)
  {
    for (j=0; j<chunk; j++)
    {
      distance = find_distance(i, j, obs, new_obs);
      check(i, j, distance, dis, ind, count);
    }
  }
}

double find_distance(int i, int j, double **obs, double **new_obs)
{
  double distance = 0;
  for (int l=0; l<dimensions; l++)
  {
    distance = distance + pow(obs[i][l]-new_obs[j][l],2);
  }
  distance = sqrt(distance);
  return (distance);
}

void check(int i, int j, double distance, double **dis, int **ind, int count)
{
  int l = 0;
  int flag = 1;
  while (l < k && flag)
  {
    if (distance < dis[i][l])
    {
      //Push the distances and indexes right to keep an ascending order.
      for (int n=k-1; n>l; n--)
      {
        dis[i][n] = dis[i][n-1];
        ind[i][n] = ind[i][n-1];
      }
      dis[i][l] = distance;
      int offset = prev - count;
      if (offset < 0)
      {
        offset = offset+numtasks;
      }
      ind[i][l] = j+1+(offset * chunk);
      flag = 0;
    }
    l++;
  }
}

/** Every process is using this function to perform a knn search
    at it's own block of data.
**/
void self_knn_search(double **obs, double **dis, int **ind)
{
  int i,j;
  double distance;
  for (i=0; i<chunk; i++)
  {
    for (j=i+1; j<chunk; j++)
    {
      distance = self_find_distance(i, j, obs);
      self_check(i, j, distance, dis, ind);
      self_check(j, i, distance, dis, ind);
    }
  }
}

//Function to find the distance between i and j points.
double self_find_distance(int i, int j, double **obs)
{
  double distance = 0;
  for (int l=0; l<dimensions; l++)
  {
    distance = distance + pow(obs[i][l]-obs[j][l],2);
  }
  distance = sqrt(distance);
  return (distance);
}

/** This function checks if the distance is smaller than
    the distances previously registered in the table of distances
    of the point i. If so, the new distance and it's index are
    being stored to the appropriate place.
**/
void self_check(int i, int j, double distance, double **dis, int **ind)
{
  int l = 0;
  int flag = 1;
  while (l < k && flag)
  {
    if (distance < dis[i][l])
    {
      //Push the distances and indexes right to keep an ascending order.
      for (int n=k-1; n>l; n--)
      {
        dis[i][n] = dis[i][n-1];
        ind[i][n] = ind[i][n-1];
      }
      dis[i][l] = distance;
      ind[i][l] = j+1+(rank * chunk);
      flag = 0;
    }
    l++;
  }
}

//Function to initialize the distances' table with a big number
void init_dis(double **a)
{
  int i,j;
  for (i=0; i<chunk; i++)
  {
    for (j=0; j<k; j++)
    {
      a[i][j] = MAXD;
    }
  }
}

/** Every process is using this function to find the labels
    at it's own block of data.
**/
void find_labels(int **lab, int *new_labels, int **ind, int count)
{
  int i,j;
  int offset = prev-count;
  if (offset < 0)
  {
    offset = offset+numtasks;
  }
  for (i=0; i<chunk; i++)
  {
    for (j=0; j<k; j++)
    {
      if (ind[i][j]>(offset*chunk) && ind[i][j]<=((offset+1)*chunk))
      {
        lab[i][j] = new_labels[ind[i][j]-1-(offset*chunk)];
      }
    }
  }
}

void self_find_labels(int **lab, int *labels, int **ind)
{
  int i,j;

  for (i=0; i<chunk; i++)
  {
    for (j=0; j<k; j++)
    {
      if (ind[i][j]>(rank*chunk) && ind[i][j]<=((rank+1)*chunk))
      {
        lab[i][j] = labels[ind[i][j]-1-(rank*chunk)];
      }
    }
  }
}

int matching(int **lab, int *labels)
{
  int i, sum = 0;
  for (i=0; i<chunk; i++)
  {
    if (labels[i] == most_common(i, lab))
    {
      sum++;
    }
  }
  float matches = (float)sum / (float)chunk * 100;
  return sum;
}

//Returns the most common element in the i row of table lab.
int most_common(int i, int **lab)
{
  //Create an array with the # of appeances of each label.
  int *app = (int *)malloc(NUML * sizeof(int));
  int j;
  //Initialize as 0
  for (j=0; j<NUML; j++)
  {
    app[j] = 0;
  }
  for (j=0; j<k; j++)
  {
    app[lab[i][j]-1]++;
  }
  //Find the max
  int max = 0;
  int com;
  for (j=0; j<NUML; j++)
  {
    if (app[j] > max)
    {
      max = app[j];
      com = j + 1;
    }
  }
  free(app);
  return(com);
}

void show_results(double **dis, int **ind)
{
  int i,j;
  for (i=0; i<5; i++)
  {
    for(j=0; j<k; j++)
    {
      printf("%f,%d ", dis[i][j], ind[i][j]);
    }
    printf("\n");
  }
}

void show_labels(int **lab)
{
  int i, j;
  for (i=0; i<5; i++)
  {
    for (j=0; j<(k+1); j++)
    {
      printf("%d ", lab[i][j]);
    }
    printf("\n");
  }
}

/** In this function every process checks how many errors
    are there in the table of indexes, using as reference a
    txt file that was produced from the solution of the
    same problem in matlab. Then it returns the number of
    errors to be summed up from the master process(rank = 0).
**/
int testing(int **ind, char *filename)
{
  FILE *fin;
  char *str = (char *)malloc(2 * k * sizeof(int));
  char *token = (char *)malloc(sizeof(int));
  int **testing = alloc_2d_int(chunk, k);
  fin = fopen(filename, "r");
  if (fin == NULL)
  {
    printf("Error opening the file...");
    exit(1);
  }
  int i = 0;
  int j;
  //Skip lines of the files if needed to store the right data.
  for (j=0; j<(rank * chunk); j++)
  {
    str = fgets(str, 2 * k * sizeof(int), fin);
  }
  str = fgets(str, 2 * k * sizeof(int), fin); //get a line of the txt file, which refers to one point.
  while (str != NULL && i < chunk)
  {
    token = strtok(str, "\t"); //get one index per recursion.
    j = 0;
    while (token != NULL && j < k)
    {
      testing[i][j] = atoi(token);
      token = strtok(NULL, "\t");
      j++;
    }
    str = fgets(str, 2 * k * sizeof(int), fin);
    i++;
  }
  fclose(fin);
  free(str);
  free(token);
  int num_errors = 0;
  for (i=0; i<chunk; i++)
  {
    for (j=0; j<k; j++)
    {
      if (testing[i][j] != ind[i][j])
      {
        num_errors++;
      }
    }
  }
  free_int(testing);
  return (num_errors);
}

//Allocating contiguous memory for a 2d table of integers.
int **alloc_2d_int(int rows, int cols) {
    int *data = (int *)malloc(rows*cols*sizeof(int));
    int **array= (int **)malloc(rows*sizeof(int*));
    for (int i=0; i<rows; i++)
    {
      array[i] = &(data[cols*i]);
    }
    return array;
}

//Allocating contiguous memory for a 2d table of doubles.
double **alloc_2d_d(int rows, int cols) {
    double *data = (double *)malloc(rows*cols*sizeof(double));
    double **array= (double **)malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++)
    {
      array[i] = &(data[cols*i]);
    }
    return array;
}

void free_int(int **a)
{
  free(a[0]);
  free(a);
}

void free_d(double **a)
{
  free(a[0]);
  free(a);
}
