/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <stdio.h> //this library is needed for printf function
#include <stdlib.h> //this library is needed for rand() function
#include <windows.h> //this library is needed for pause() function
#include <time.h> //this library is needed for clock() function
#include <math.h> //this library is needed for abs()
#include <pmmintrin.h>
#include <process.h>
//#include <chrono>
#include <iostream>
#include <immintrin.h>

void initialize();
void initialize_again();
void slow_routine(float alpha, float beta);//you will optimize this routine
unsigned short int Compare(float alpha, float beta);
unsigned short int equal(float const a, float const b);

#define N 8192 //input size
__declspec(align(64)) float A[N][N], u1[N], u2[N], v1[N], v2[N], x[N], y[N], w[N], z[N], test[N];

#define TIMES_TO_RUN 1 //how many times the function will run
#define EPSILON 0.0001

int main() {

	float alpha = 0.23f, beta = 0.45f;

	//define the timers measuring execution time
	clock_t start_1, end_1; //ignore this for  now

	initialize();

	start_1 = clock(); //start the timer 

	for (int i = 0; i < TIMES_TO_RUN; i++)//this loop is needed to get an accurate ex.time value
		slow_routine(alpha, beta);


	end_1 = clock(); //end the timer 

	printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));//print the ex.time

	if (Compare(alpha, beta) == 0)
		printf("\nCorrect Result\n");
	else
		printf("\nINcorrect Result\n");
	float FLOPS = (12 * N ^ 2 + 2 * N);
	printf("\n%f FLOPS achieved\n", FLOPS);
	system("pause"); //this command does not let the output window to close
	
	return 0; //normally, by returning zero, we mean that the program ended successfully. 
}


void initialize() {

	unsigned int    i, j;

	//initialization
	for (i = 0;i < N;i++)
		for (j = 0;j < N;j++) {
			A[i][j] = (i % 21) + (j % 32) - 0.012f;

		}

	for (i = 0;i < N;i++) {
		z[i] = (i % 9) * 0.8f;
		x[i] = 0.1f;
		u1[i] = (i % 9) * 0.2f;
		u2[i] = (i % 9) * 0.3f;
		v1[i] = (i % 9) * 0.4f;
		v2[i] = (i % 9) * 0.5f;
		w[i] = 0.0f;
		y[i] = (i % 9) * 0.7f;
	}

}

void initialize_again() {

	unsigned int    i, j;

	//initialization
	for (i = 0;i < N;i++)
		for (j = 0;j < N;j++) {
			A[i][j] = (i % 21) + (j % 32) - 0.012f;

		}

	for (i = 0;i < N;i++) {
		z[i] = (i % 9) * 0.8f;
		x[i] = 0.1f;
		test[i] = 0.0f;
		u1[i] = (i % 9) * 0.2f;
		u2[i] = (i % 9) * 0.3f;
		v1[i] = (i % 9) * 0.4f;
		v2[i] = (i % 9) * 0.5f;
		y[i] = (i % 9) * 0.7f;
	}

}

//you will optimize this routine
void slow_routine(float alpha, float beta) {

	unsigned int i, j;
	
	//transpose this
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			A[i][j] += alpha * u1[i] * v1[j] + u2[i] * v2[j];

	// transpose this
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			x[i] += A[j][i] * y[j] + beta;


	// Need to add loop for extra iterations (e.g., if N = 10 need to account for the 2 extra iterations)
	__m256 ymm0, ymm1, ymm2, ymm3;
	ymm0 = _mm256_set_ps(3.22f, 3.22f, 3.22f, 3.22f, 3.22f, 3.22f, 3.22f, 3.22f); // set num values

	for (i = 0; i < N; i += 8)
		ymm3 = _mm256_load_ps(&z[i]); // load 8 elements of z[]
		ymm1 = _mm256_load_ps(&x[i]); // load 8 elements of x[]
		_mm256_fmadd_ps(ymm0, ymm3, ymm0); // multiply ymm0 and ymm3, store results into ymm0
		ymm2 = _mm256_add_ps(ymm1, ymm0); // add ymm1 and ymm0, store results into ymm2
		_mm256_store_ps(&x[i], ymm2); // store values of ymm2 into x[]


	__m256 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, znum;
	float temp[8];
	zmm0 = _mm256_set_ps(alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha); // set num values
	zmm1 = _mm256_set_ps(beta, beta, beta, beta, beta, beta, beta, beta);
	znum = _mm256_setzero_ps();
	for (i = 0; i < N; i++)
		zmm5 = _mm256_setzero_ps();
		zmm6 = _mm256_setzero_ps();
		for (j = 0; j < N; j += 8)
			zmm2 = _mm256_load_ps(&A[i][j]); // load 8 elements of A[][]
			zmm3 = _mm256_load_ps(&x[j]); // load 8 elements of x[]
			_mm256_fmadd_ps(zmm0, zmm2, znum); // multiply zmm0 (alpha) and zmm2 (A[][]), store results in znum
			_mm256_fmadd_ps(znum, zmm3, znum); // multiply znum (alpha * A[][]) and zmm3 (x[]), store results into znum
			zmm5 = _mm256_add_ps(znum, zmm1); // add znum (alpha * A[][] * x[]) and zmm1 (beta), store results into zmm5
			zmm6 = _mm256_add_ps(zmm5, zmm6); // add zmm5 (alpha * A[][] * x[] + beta) and zmm6 (acts as total), store results into zmm6
			// w[i] += alpha * A[i][j] * x[j] + beta;
			
		zmm6 = _mm256_hadd_ps(zmm6, zmm6); // pack zmm6 into one value for total
		zmm6 = _mm256_hadd_ps(zmm6, zmm6);
		_mm256_storeu_ps(temp, zmm6);
		w[i] += temp[0];
			


}


unsigned short int Compare(float alpha, float beta) {

	unsigned int i, j;

	initialize_again();


	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			A[i][j] += alpha * u1[i] * v1[j] + u2[i] * v2[j];


	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			x[i] += A[j][i] * y[j] + beta;

	for (i = 0; i < N; i++)
		x[i] += 3.22f * z[i];


	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			test[i] += alpha * A[i][j] * x[j] + beta;


	for (j = 0; j < N; j++) {
		if (equal(w[j], test[j]) == 1) {
			printf("\n %f %f", test[j], w[j]);
			return -1;
		}
	}

	return 0;
}




unsigned short int equal(float const a, float const b) {

	if (fabs(a - b) / fabs(a) < EPSILON)
		return 0; //success
	else
		return 1;
}


