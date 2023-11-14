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
	float FLOPS = (12 * N * N + 2 * N);
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

	
	__m256 wmm0, wmm1, wmm2, wmm3, wmm4, wmm5, wmm6, wmmalpha;
	wmmalpha = _mm256_set1_ps(alpha); // 8 copies of alpha
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j += 8) {
			wmm0 = _mm256_set1_ps(u1[i]); // 8 copies of u1[i]
			wmm1 = _mm256_set1_ps(u2[i]); // 8 copies of u2[i]
			wmm5 = _mm256_mul_ps(wmmalpha, wmm0); // alpha * u1[i]
			wmm2 = _mm256_load_ps(&A[i][j]); // load 8 elements of A[i][]
			wmm3 = _mm256_load_ps(&v1[j]); // load 8 elements of v1[]
			wmm4 = _mm256_load_ps(&v2[j]); // load 8 elements of v2[]
			wmm3 = _mm256_mul_ps(wmm3, wmm5); // v1[] * (alpha * u1[i])
			wmm1 = _mm256_mul_ps(wmm1, wmm4); // u2[i] * v2[]
			wmm3 = _mm256_add_ps(wmm3, wmm1); // ((alpha * u1[i]) * v1[]) + (u2[i] * v2[])
			wmm3 = _mm256_add_ps(wmm2, wmm3); // A[][] + (((alpha * u1[i]) * v1[]) + (u2[i] * v2[]))

			_mm256_store_ps(&A[i][j], wmm3);

			// A[i][j] += alpha * u1[i] * v1[j] + u2[i] * v2[j];
			// A[i][j] = A[i][j] + alpha * u1[i] * v1[j] + u2[i] * v2[j] 
		}
	}
	

	
	__m256 xmm0, xmm1, xmm2, xmm3, xmmbeta;

	xmmbeta = _mm256_set1_ps(beta); // load 8 copies of beta

	for (j = 0; j < N; j++) {
		xmm0 = _mm256_set1_ps(y[j]); // load 8 copies of y[]
		for (i = 0; i < N; i+=8) {
			xmm1 = _mm256_load_ps(&x[i]); // load 8 elements of x[]
			xmm2 = _mm256_load_ps(&A[j][i]); // load 8 elements of A[j][]

			xmm3 = _mm256_mul_ps(xmm2, xmm0); // A[][] * y[j]
			xmm3 = _mm256_add_ps(xmm3, xmmbeta); // (A[][] * y[j]) + beta
			xmm3 = _mm256_add_ps(xmm3, xmm1); // ((A[][] * y[j]) + beta) + x[]

			_mm256_store_ps(&x[i], xmm3); // store xmm3 into x[]


		}
	}
		


	// Need to add loop for extra iterations (e.g., if N = 10 need to account for the 2 extra iterations)
	__m256 ymm0, ymm1, ymm2, ymm3;
	ymm0 = _mm256_set_ps(3.22f, 3.22f, 3.22f, 3.22f, 3.22f, 3.22f, 3.22f, 3.22f); // set num values

	for (i = 0; i < N; i += 8)
		ymm3 = _mm256_load_ps(&z[i]); // load 8 elements of z[]
		ymm1 = _mm256_load_ps(&x[i]); // load 8 elements of x[]
		_mm256_fmadd_ps(ymm0, ymm3, ymm0); // multiply ymm0 and ymm3, store results into ymm0
		ymm2 = _mm256_add_ps(ymm1, ymm0); // add ymm1 and ymm0, store results into ymm2
		_mm256_store_ps(&x[i], ymm2); // store values of ymm2 into x[]
	
	__m256 zmm0, zmm1, zmm2, zmm3, zmmalpha, zmmbeta;
	__m128 total;
	for (i = 0; i < N; i++) {
		zmm0 = _mm256_set1_ps(w[i]); // 8 copies of w[i]
		zmm3 = _mm256_setzero_ps();
		zmmalpha = _mm256_set1_ps(alpha); // 8 copies of alpha
		zmmbeta = _mm256_set1_ps(beta); //8 copies of beta
		for (j = 0; j < N; j += 8) {
			zmm1 = _mm256_load_ps(&A[i][j]);
			zmm2 = _mm256_load_ps(&x[j]);
			zmm1 = _mm256_mul_ps(zmm1, zmm2); // A[][] * x[]
			zmm1 = _mm256_mul_ps(zmm1, zmmalpha); // (A[][] * x[]) * alpha
			zmm1 = _mm256_add_ps(zmm1, zmmbeta); // ((A[][] * x[]) * alpha) + beta
			zmm3 = _mm256_add_ps(zmm3, zmm1);
		}
		
		zmm0 = _mm256_permute2f128_ps(zmm3, zmm3, 1);
		zmm3 = _mm256_add_ps(zmm3, zmm0);
		zmm3 = _mm256_hadd_ps(zmm3, zmm3); // pack zmm1 into one value for total
		zmm3 = _mm256_hadd_ps(zmm3, zmm3);
		total = _mm256_extractf128_ps(zmm3, 0); // extract packed zmm1
		_mm_store_ss(&w[i], total); // store total into w[i]
	}
		

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


