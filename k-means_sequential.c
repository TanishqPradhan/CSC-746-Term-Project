/*
Name: Tanishq Pradhan
Student ID: 922243982
CSC 746 Term Project
Fall 2022
*/

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define MAX_ITER 100
#define THRESHOLD 1e-6
#define min(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

int numberOfPointsGlobal;
int numberOfIterationsGlobal;
double deltaGlobal = THRESHOLD + 1;
int KGlobal;
int *dataPointsGlobal;
float *iterCentroidsGlobal;
int *dataPointsClusterGlobal;

void kmeans_sequential_execution()
{
    printf("Sequential k-means start\n");

    int i = 0, j = 0;
    double min_dist, current_dist;

	// Cluster id associated with each point
    int *point_to_cluster_id = (int *)malloc(numberOfPointsGlobal * sizeof(int));

	// Cluster location or centroid (x,y,z) coordinates for K clusters in an iteration
    float *cluster_points_sum = (float *)malloc(KGlobal * 3 * sizeof(float));

	// Number of points in a cluster for an iteration
    int *points_inside_cluster_count = (int *)malloc(KGlobal * sizeof(int));

	// Loop start
    int iter_counter = 0;
    double temp_delta = 0.0;
    while ((deltaGlobal > THRESHOLD) && (iter_counter < MAX_ITER)) //+1 is for the last assignment to cluster centroids (from previous iter)
    {
		// Initialize cluster_points_sum or centroid to 0.0
        for (i = 0; i < KGlobal * 3; i++)
            cluster_points_sum[i] = 0.0;

		// Initialize number of points for each cluster to 0
        for (i = 0; i < KGlobal; i++)
            points_inside_cluster_count[i] = 0;

        for (i = 0; i < numberOfPointsGlobal; i++)
        {
            //Assign these points to their nearest cluster
            min_dist = DBL_MAX;
            for (j = 0; j < KGlobal; j++)
            {
                current_dist = pow((double)(iterCentroidsGlobal[(iter_counter * KGlobal + j) * 3] - (float)dataPointsGlobal[i * 3]), 2.0) +
                               pow((double)(iterCentroidsGlobal[(iter_counter * KGlobal + j) * 3 + 1] - (float)dataPointsGlobal[i * 3 + 1]), 2.0) +
                               pow((double)(iterCentroidsGlobal[(iter_counter * KGlobal + j) * 3 + 2] - (float)dataPointsGlobal[i * 3 + 2]), 2.0);
                if (current_dist < min_dist)
                {
                    min_dist = current_dist;
                    point_to_cluster_id[i] = j;
                }
            }

             //Update local count of number of points inside cluster
            points_inside_cluster_count[point_to_cluster_id[i]] += 1;

			// Update local sum of cluster data points
            cluster_points_sum[point_to_cluster_id[i] * 3] += (float)dataPointsGlobal[i * 3];
            cluster_points_sum[point_to_cluster_id[i] * 3 + 1] += (float)dataPointsGlobal[i * 3 + 1];
            cluster_points_sum[point_to_cluster_id[i] * 3 + 2] += (float)dataPointsGlobal[i * 3 + 2];
        }

        //Compute centroid from cluster_points_sum and store inside iterCentroidsGlobal in a iteration
        for (i = 0; i < KGlobal; i++)
        {
            assert(points_inside_cluster_count[i] != 0);
            iterCentroidsGlobal[((iter_counter + 1) * KGlobal + i) * 3] = cluster_points_sum[i * 3] / points_inside_cluster_count[i];
            iterCentroidsGlobal[((iter_counter + 1) * KGlobal + i) * 3 + 1] = cluster_points_sum[i * 3 + 1] / points_inside_cluster_count[i];
            iterCentroidsGlobal[((iter_counter + 1) * KGlobal + i) * 3 + 2] = cluster_points_sum[i * 3 + 2] / points_inside_cluster_count[i];
        }

	/*
    	Delta is the sum of squared distance between centroid of previous and current iteration.
    	Supporting formula is:
        	delta = (iter1_centroid1_x - iter2_centroid1_x)^2 + (iter1_centroid1_y - iter2_centroid1_y)^2 + (iter1_centroid1_z - iter2_centroid1_z)^2 + (iter1_centroid2_x - iter2_centroid2_x)^2 + (iter1_centroid2_y - iter2_centroid2_y)^2 + (iter1_centroid2_z - iter2_centroid2_z)^2
    	Update deltaGlobal with new delta
	*/
        temp_delta = 0.0;
        for (i = 0; i < KGlobal; i++)
        {
            temp_delta += (iterCentroidsGlobal[((iter_counter + 1) * KGlobal + i) * 3] - iterCentroidsGlobal[((iter_counter)*KGlobal + i) * 3]) * (iterCentroidsGlobal[((iter_counter + 1) * KGlobal + i) * 3] - iterCentroidsGlobal[((iter_counter)*KGlobal + i) * 3]) + (iterCentroidsGlobal[((iter_counter + 1) * KGlobal + i) * 3 + 1] - iterCentroidsGlobal[((iter_counter)*KGlobal + i) * 3 + 1]) * (iterCentroidsGlobal[((iter_counter + 1) * KGlobal + i) * 3 + 1] - iterCentroidsGlobal[((iter_counter)*KGlobal + i) * 3 + 1]) + (iterCentroidsGlobal[((iter_counter + 1) * KGlobal + i) * 3 + 2] - iterCentroidsGlobal[((iter_counter)*KGlobal + i) * 3 + 2]) * (iterCentroidsGlobal[((iter_counter + 1) * KGlobal + i) * 3 + 2] - iterCentroidsGlobal[((iter_counter)*KGlobal + i) * 3 + 2]);
        }
        deltaGlobal = temp_delta;

        iter_counter++;
    }

	// Store the number of iterations performed in global variable
    numberOfIterationsGlobal
 = iter_counter;

    // Assign points to final choice for cluster centroids
    for (i = 0; i < numberOfPointsGlobal; i++)
    {
        // Assign points to clusters
        dataPointsClusterGlobal[i * 4] = dataPointsGlobal[i * 3];
        dataPointsClusterGlobal[i * 4 + 1] = dataPointsGlobal[i * 3 + 1];
        dataPointsClusterGlobal[i * 4 + 2] = dataPointsGlobal[i * 3 + 2];
        dataPointsClusterGlobal[i * 4 + 3] = point_to_cluster_id[i];
        assert(point_to_cluster_id[i] >= 0 && point_to_cluster_id[i] < KGlobal);
    }
}

void kmeans_sequential(int N,
					int K,
					int* data_points,
					int** data_point_cluster_id,
					float** iter_centroids,
					int* num_iterations
					)
{

    // Initialize global variables
    numberOfPointsGlobal = N;
    numberOfIterationsGlobal
 = *num_iterations;
    KGlobal = K;
    dataPointsGlobal = data_points;

	//Allocating space of 4 units each for N data points
    *data_point_cluster_id = (int *)malloc(N * 4 * sizeof(int));
    dataPointsClusterGlobal = *data_point_cluster_id;

    /*
        Allocating space of 3K units for each iteration
        Since three dimensional data point and K number of clusters 
    */
    iterCentroidsGlobal = (float *)calloc((MAX_ITER + 1) * K * 3, sizeof(float));

    // Assign first K points to be initial centroids
    int i = 0;
    for (i = 0; i < K; i++)
    {
        iterCentroidsGlobal[i * 3] = data_points[i * 3];
        iterCentroidsGlobal[i * 3 + 1] = data_points[i * 3 + 1];
        iterCentroidsGlobal[i * 3 + 2] = data_points[i * 3 + 2];
    }

    // Print initial centroids
    for (i = 0; i < K; i++)
    {
        printf("initial centroid #%d: %f,%f,%f\n", i + 1, iterCentroidsGlobal[i * 3], iterCentroidsGlobal[i * 3 + 1], iterCentroidsGlobal[i * 3 + 2]);
    }

    // Run k-means sequential function
    kmeans_sequential_execution();

    // Record number of iterations and store iterCentroidsGlobal data into iter_centroids
    *num_iterations = numberOfIterationsGlobal
;
    int centroids_size = (*num_iterations + 1) * K * 3;
    printf("number of iterations:%d\n", numberOfIterationsGlobal
);
    *iter_centroids = (float *)calloc(centroids_size, sizeof(float));
    for (i = 0; i < centroids_size; i++)
    {
        (*iter_centroids)[i] = iterCentroidsGlobal[i];
    }

    // Print final centroids
    for (i = 0; i < K; i++)
    {
        printf("centroid #%d: %f,%f,%f\n", i + 1, (*iter_centroids)[((*num_iterations) * K + i) * 3], (*iter_centroids)[((*num_iterations) * K + i) * 3 + 1], (*iter_centroids)[((*num_iterations) * K + i) * 3 + 2]);
    }
}

void dataset_in(const char *dataset_filename, int *N, int **data_points)
{
	FILE *fin = fopen(dataset_filename, "r");
	fscanf(fin, "%d", N);
	*data_points = (int *)malloc(sizeof(int) * ((*N) * 3));
    int i = 0;
	for (i = 0; i < (*N) * 3; i++)
	{
		fscanf(fin, "%d", (*data_points + i));
	}
	fclose(fin);
}

void clusters_out(const char *cluster_filename, int N, int *cluster_points)
{
	FILE *fout = fopen(cluster_filename, "w");
    int i = 0;
	for (i = 0; i < N; i++)
	{
		fprintf(fout, "%d %d %d %d\n",
				*(cluster_points + (i * 4)), *(cluster_points + (i * 4) + 1),
				*(cluster_points + (i * 4) + 2), *(cluster_points + (i * 4) + 3));
	}
	fclose(fout);
}

void centroids_out(const char *centroid_filename, int K, int number_of_iterations, float *iter_centroids)
{
	FILE *fout = fopen(centroid_filename, "w");
    int i = 0;
	for (i = 0; i < number_of_iterations + 1; i++)
	{
        int j = 0;
		for (j = 0; j < K; j++)
		{
			fprintf(fout, "%f %f %f, ",
					*(iter_centroids + (i * K + j) * 3),		 //x coordinate
					*(iter_centroids + (i * K + j) * 3 + 1),  //y coordinate
					*(iter_centroids + (i * K + j) * 3 + 2)); //z coordinate
		}
		fprintf(fout, "\n");
	}
	fclose(fout);
}

int main()
{

	//---------------------------------------------------------------------
	int N;					// Number of data points (input)
	int K;					//Number of clusters to be formed (input)
	int* data_points;		//Data points (input)
	int* cluster_points;	//clustered data points 
	float* iter_centroids;		//centroids of each iteration 
	int number_of_iterations;     //no of iterations performed by algo 
	//---------------------------------------------------------------------

	double start_time, end_time;
	double computation_time;

	printf("Enter No. of Clusters: ");
    scanf("%d", &K);

	printf("\nFollowing files should be in the same directory as of program\n");
    printf("1 for 10000 datapoints\n");
    printf("2 for 50000 datapoints\n");
    printf("3 for 100000 datapoints\n");
    printf("4 for 500000 datapoints\n");
    printf("5 for 1000000 datapoints\n");
    printf("\nEnter the number of dataset file to input: ");

    int x;
    scanf("%d",&x);

	char *dataset_filename = "dataset-10000.txt";

    switch (x)
    {
    case 1:
        dataset_filename = "dataset-10000.txt";
        break;
    case 2:
        dataset_filename = "dataset-50000.txt";
        break;
    case 3:
        dataset_filename = "dataset-100000.txt";
        break;
    case 4:
        dataset_filename = "dataset-500000.txt";
        break;
    case 5:
        dataset_filename = "dataset-1000000.txt";
        break;
    default:
        dataset_filename = "dataset-10000.txt";
        break;
    }


	/*
        Function reads dataset_file and store data into data_points array. Each points have three consecutive indices associated into array.
        data_points array looks like : [pt_1_x, pt_1_y, pt_1_z, pt_2_x, pt_2_y, pt_2_z]
	*/
	dataset_in (dataset_filename, &N, &data_points);

	start_time = omp_get_wtime();
	kmeans_sequential(N, K, data_points, &cluster_points, &iter_centroids, &number_of_iterations);
	end_time = omp_get_wtime();	

	// Creating filenames for different dataset

    char file_index_char[2];
    snprintf(file_index_char,10,"%d", x);

    char cluster_filename[105] = "cluster_output_dataset";
    strcat(cluster_filename,file_index_char);
    strcat(cluster_filename,".txt");

    char centroid_filename[105] = "centroid_output_dataset";
    strcat(centroid_filename,file_index_char);
    strcat(centroid_filename,".txt");

	/*
        Clustered points are saved into cluster_filename.
        Each point is associated with the cluster index it belongs to.
        cluster_points array looks like : [pt_1_x, pt_1_y, pt_1_z, pt_1_cluster_index, pt_2_x, pt_2_y, pt_2_z, pt_2_cluster_index]
        Output file format:
            pt_1_x, pt_1_y, pt_1_z, pt_1_cluster_index
	*/
	clusters_out (cluster_filename, N, cluster_points);

	/*
        Centroid points are stored into centroid_filename.
        Each line in the file depicts the centroid coordinates of clusters after each iteration.
        Output file format:
            centroid_1_x, centroid_1_y, centroid_1_z, centroid_2_x, centroid_2_y, centroid_2_z
    */
	centroids_out (centroid_filename, K, number_of_iterations, iter_centroids);

	/*
        Computation time is stored in 'compute_time_openmp.txt'.
    */
   	computation_time = end_time - start_time;
	printf("Time Taken: %lf \n", computation_time);
    
	char time_file_omp[100] = "compute_time_openmp_dataset";
    strcat(time_file_omp,file_index_char);
    strcat(time_file_omp,".txt");

	FILE *fout = fopen(time_file_omp, "a");
	fprintf(fout, "%f\n", computation_time);
	fclose(fout);
    
	printf("Cluster Centroid point output file '%s' saved\n", centroid_filename);
    printf("Clustered points output file '%s' saved\n", cluster_filename);
    printf("Computation time output file '%s' saved\n", time_file_omp);
	
	return 0;
}