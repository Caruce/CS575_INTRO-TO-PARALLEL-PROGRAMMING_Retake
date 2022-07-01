#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdlib>



#define NUMT 4
unsigned int seed = 0;

omp_lock_t	Lock;
int		NumInThreadTeam;
int		NumAtBarrier;
int		NumGone;

int	NowYear = 2021;		// 2021 - 2026
int	NowMonth;		// 0 - 11

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight = 1;		// grain height in inches
int	    NowNumDeer = 10;		// number of deer in the current population
int     CatchByHunter = 0;

const float GRAIN_GROWS_PER_MONTH =		9.0;
const float ONE_DEER_EATS_PER_MONTH =		1.0;

const float AVG_PRECIP_PER_MONTH =		7.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise

const float AVG_TEMP =				60.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;


void	InitBarrier(int);
void	WaitBarrier();


float
Ranf( unsigned int *seedp,  float low, float high )
{
    float r = (float) rand_r( seedp );              // 0 - RAND_MAX

    return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}


int
Ranf( unsigned int *seedp, int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = (float)ihigh + 0.9999f;

        return (int)(  Ranf(seedp, low,high) );
}

float
SQR(float x)
{
	return x * x;
}

void Grain()
{
	float nextHeight;
	float tempFactor;
	float precipFactor;

	while (NowYear < 2027)
	{
		// DoneComputing barrier:
		tempFactor = exp(-SQR((NowTemp - MIDTEMP) / 10.));
		precipFactor = exp(-SQR((NowPrecip - MIDPRECIP) / 10.));
		nextHeight = NowHeight;
		nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
		if (nextHeight < 0)
			nextHeight = 0.;
		WaitBarrier();

		// DoneAssigning barrier:
		NowHeight = nextHeight;
		WaitBarrier();

		// DonePrinting barrier:
		WaitBarrier();
	}
}

void GrainDeer()
{
	int nextNumDeer;
	while (NowYear < 2027)
	{
		// DoneComputing barrier:
		int nextNumDeer = NowNumDeer;
        int carryingCapacity = (int)( NowHeight );
        if( nextNumDeer < carryingCapacity )
            nextNumDeer++;
        else
         if( nextNumDeer > carryingCapacity )
                nextNumDeer--;

        if( nextNumDeer < 0 )
            nextNumDeer = 0;
		WaitBarrier();

		// DoneAssigning barrier:
		NowNumDeer = nextNumDeer;
		WaitBarrier();

		// DonePrinting barrier:
		WaitBarrier();
	}
}

void Hunter()
{
	int nextNumDeer;
	while (NowYear < 2027)
	{
		// DoneComputing barrier:
		if (NowNumDeer >= 6)
		{
			nextNumDeer = (int)NowNumDeer - 1;
			CatchByHunter++;
		}
		else 
			nextNumDeer = NowNumDeer;
		WaitBarrier();

		// DoneAssigning barrier:
		NowNumDeer = nextNumDeer;
		WaitBarrier();

		// DonePrinting barrier:
		WaitBarrier();
	}
}

void Watcher()
{
	while (NowYear < 2027)
	{

		// DoneComputing barrier:
		WaitBarrier();

		// DoneAssigning barrier:
		WaitBarrier();

		printf("%d\t%d\t%.2f\t%.2f\t%.2f\t%d\t%d\n", NowYear, (NowMonth + 1), NowTemp, NowPrecip, NowHeight, NowNumDeer, CatchByHunter);
		// DonePrinting barrier:
		float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.);
		float temp = AVG_TEMP - AMP_TEMP * cos(ang);
		NowTemp = temp + Ranf(&seed, -RANDOM_TEMP, RANDOM_TEMP);
		float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
		NowPrecip = precip + Ranf(&seed, -RANDOM_PRECIP, RANDOM_PRECIP);
		if (NowPrecip < 0.)
			NowPrecip = 0.;

		if ((NowMonth + 1) != 12)
			NowMonth++;
		else {
			NowMonth = 0;
			NowYear++;
		}
		WaitBarrier();
	}
}

int main()
{
	InitBarrier(4);
	omp_set_num_threads(4);	// same as # of sections
	printf("Year\tMonth\tTemp\tPrec\tHeight\tNumber of Deer\tCatchByHunter\n");

#pragma omp parallel sections
	{
#pragma omp section
		{
			GrainDeer();
		}

#pragma omp section
		{
			Grain();
		}

#pragma omp section
		{
			Watcher();
		}

#pragma omp section
		{
			Hunter();	// your own
		}
	}
}







// specify how many threads will be in the barrier:
//	(also init's the Lock)

void
InitBarrier( int n )
{
        NumInThreadTeam = n;
        NumAtBarrier = 0;
	omp_init_lock( &Lock );
}


// have the calling thread wait here until all the other threads catch up:

void
WaitBarrier( )
{
        omp_set_lock( &Lock );
        {
                NumAtBarrier++;
                if( NumAtBarrier == NumInThreadTeam )
                {
                        NumGone = 0;
                        NumAtBarrier = 0;
                        // let all other threads get back to what they were doing
			// before this one unlocks, knowing that they might immediately
			// call WaitBarrier( ) again:
                        while( NumGone != NumInThreadTeam-1 );
                        omp_unset_lock( &Lock );
                        return;
                }
        }
        omp_unset_lock( &Lock );

        while( NumAtBarrier != 0 );	// this waits for the nth thread to arrive

        #pragma omp atomic
        NumGone++;			// this flags how many threads have returned
}