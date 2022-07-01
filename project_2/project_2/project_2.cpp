#define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#ifndef NUMT
#define NUMT		8
#endif

#ifndef NUMNODES
#define NUMNODES		4
#endif
#define N	0.70


float
Height(int iu, int iv)	// iu,iv = 0 .. NUMNODES-1
{
    float x = -1. + 2. * (float)iu / (float)(NUMNODES - 1);	// -1. to +1.
    float y = -1. + 2. * (float)iv / (float)(NUMNODES - 1);	// -1. to +1.

    float xn = pow(fabs(x), (double)N);
    float yn = pow(fabs(y), (double)N);
    float r = 1. - xn - yn;
    if (r <= 0.)
        return 0.;
    float height = pow(r, 1. / (float)N);
    return height;
}


#define XMIN     -1.
#define XMAX      1.
#define YMIN     -1.
#define YMAX      1.

#define N	0.70

float Height(int, int);	// function prototype

int main(int argc, char* argv[])
{
    float maxPerformance = 0.;
    float volume = 0;
    // the area of a single full-sized tile:
    // (not all tiles are full-sized, though)

    float fullTileArea = (((XMAX - XMIN) / (float)(NUMNODES - 1)) *
        ((YMAX - YMIN) / (float)(NUMNODES - 1)));

    // sum up the weighted heights into the variable "volume"
    // using an OpenMP for loop and a reduction:
    omp_set_num_threads(NUMT);

    double time0 = omp_get_wtime();
#pragma omp parallel for default(none) 

    for (int i = 0; i < NUMNODES * NUMNODES; i++)
    {
        float area = 0;
        int iu = i % NUMNODES;
        int iv = i / NUMNODES;
        float z = Height(iu, iv);
        // calculate each voloum
        if ((iu == 0 && iv == 0) || (iu == NUMNODES - 1 && iv == NUMNODES - 1) || (iu == 0 && iv == NUMNODES - 1) || (iu == NUMNODES - 1 && iv == 0))
            area = z * fullTileArea * 1 / 4;
        if (iu == 0 || iv == 0)
            area = z * fullTileArea * 1 / 2;
        else
            area = z * fullTileArea;

        volume = volume + 2 * area * Height(iu, iv);
    }
    double time1 = omp_get_wtime();

    double MegaHeightsComputedPerSecond = (double)(NUMNODES * NUMNODES) / (time1 - time0);


    printf("the number of threads = %d\n", NUMT);
    printf("the number of nodes = %d\n", NUMNODES);
    printf("the volume = %lf\n", volume);
    printf("Peak Performance = %8.2lf Heights/Sec\n", MegaHeightsComputedPerSecond);
    printf("Time costing = %8.5lf Sec\n", (time1 - time0));
}