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

	// Optimisations applied - vectorisation, loop interchange, loop merge, register blocking

	unsigned int i, j;

	
	__m256 wmm0, wmm0i1, wmm0i2, wmm0i3, wmm0i4, wmm0i5, wmm0i6, wmm0i7, wmm1, wmm1i1, wmm1i2, wmm1i3, wmm1i4, wmm1i5, wmm1i6, wmm1i7, wmm2, wmm3, wmm4, wmm5, wmm6, wmm6i1, wmm6i2, wmm6i3, wmm6i4, wmm6i5, wmm6i6, wmm6i7, wmm7, wmm8, wmm9, wmmalpha;
	wmmalpha = _mm256_set1_ps(alpha); // 8 copies of alpha
	for (i = 0; i < N; i += 8) {
		wmm0 = _mm256_set1_ps(u1[i]); // 8 copies of u1[i]
		wmm1 = _mm256_set1_ps(u2[i]); // 8 copies of u2[i]	

		wmm0i1 = _mm256_set1_ps(u1[i + 1]);
		wmm1i1 = _mm256_set1_ps(u2[i + 1]);
		
		wmm0i2 = _mm256_set1_ps(u1[i + 2]);
		wmm1i2 = _mm256_set1_ps(u2[i + 2]);
		
		wmm0i3 = _mm256_set1_ps(u1[i + 3]);
		wmm1i3 = _mm256_set1_ps(u2[i + 3]);
		
		wmm0i4 = _mm256_set1_ps(u1[i + 4]);
		wmm1i4 = _mm256_set1_ps(u2[i + 4]);
		
		wmm0i5 = _mm256_set1_ps(u1[i + 5]);
		wmm1i5 = _mm256_set1_ps(u2[i + 5]);
		
		wmm0i6 = _mm256_set1_ps(u1[i + 6]);
		wmm1i6 = _mm256_set1_ps(u2[i + 6]);
		
		wmm0i7 = _mm256_set1_ps(u1[i + 7]);
		wmm1i7 = _mm256_set1_ps(u2[i + 7]);
		for (j = 0; j < N; j += 8) {
			
			wmm2 = _mm256_load_ps(&A[i][j]); // load 8 elements of A[i][]
			wmm3 = _mm256_load_ps(&v1[j]); // load 8 elements of v1[]
			wmm4 = _mm256_load_ps(&v2[j]); // load 8 elements of v2[]
			wmm5 = _mm256_mul_ps(wmmalpha, wmm0); // alpha * u1[i]
			wmm7 = _mm256_mul_ps(wmm3, wmm5); // v1[] * (alpha * u1[i])
			wmm8 = _mm256_mul_ps(wmm1, wmm4); // u2[i] * v2[]
			wmm9 = _mm256_add_ps(wmm7, wmm8); // ((alpha * u1[i]) * v1[]) + (u2[i] * v2[])
			wmm6 = _mm256_add_ps(wmm2, wmm9); // A[][] + (((alpha * u1[i]) * v1[]) + (u2[i] * v2[]))

			// i + 1
			
			wmm2 = _mm256_load_ps(&A[i + 1][j]);
			wmm5 = _mm256_mul_ps(wmmalpha, wmm0i1);
			wmm7 = _mm256_mul_ps(wmm3, wmm5);
			wmm8 = _mm256_mul_ps(wmm1i1, wmm4);
			wmm9 = _mm256_add_ps(wmm7, wmm8);
			wmm6i1 = _mm256_add_ps(wmm2, wmm9);

			// i + 2
			
			wmm2 = _mm256_load_ps(&A[i + 2][j]);
			wmm5 = _mm256_mul_ps(wmmalpha, wmm0i2);
			wmm7 = _mm256_mul_ps(wmm3, wmm5);
			wmm8 = _mm256_mul_ps(wmm1i2, wmm4);
			wmm9 = _mm256_add_ps(wmm7, wmm8);
			wmm6i2 = _mm256_add_ps(wmm2, wmm9);

			// i + 3
			
			wmm2 = _mm256_load_ps(&A[i + 3][j]);
			wmm5 = _mm256_mul_ps(wmmalpha, wmm0i3);
			wmm7 = _mm256_mul_ps(wmm3, wmm5);
			wmm8 = _mm256_mul_ps(wmm1i3, wmm4);
			wmm9 = _mm256_add_ps(wmm7, wmm8);
			wmm6i3 = _mm256_add_ps(wmm2, wmm9);

			// i + 4
			
			wmm2 = _mm256_load_ps(&A[i + 4][j]);
			wmm5 = _mm256_mul_ps(wmmalpha, wmm0i4);
			wmm7 = _mm256_mul_ps(wmm3, wmm5);
			wmm8 = _mm256_mul_ps(wmm1i4, wmm4);
			wmm9 = _mm256_add_ps(wmm7, wmm8);
			wmm6i4 = _mm256_add_ps(wmm2, wmm9);

			// i + 5
			
			wmm2 = _mm256_load_ps(&A[i + 5][j]);
			wmm5 = _mm256_mul_ps(wmmalpha, wmm0i5);
			wmm7 = _mm256_mul_ps(wmm3, wmm5);
			wmm8 = _mm256_mul_ps(wmm1i5, wmm4);
			wmm9 = _mm256_add_ps(wmm7, wmm8);
			wmm6i5 = _mm256_add_ps(wmm2, wmm9);

			// i + 6
			
			wmm2 = _mm256_load_ps(&A[i + 6][j]);
			wmm5 = _mm256_mul_ps(wmmalpha, wmm0i6);
			wmm7 = _mm256_mul_ps(wmm3, wmm5);
			wmm8 = _mm256_mul_ps(wmm1i6, wmm4);
			wmm9 = _mm256_add_ps(wmm7, wmm8);
			wmm6i6 = _mm256_add_ps(wmm2, wmm9);

			// i + 7
			
			wmm2 = _mm256_load_ps(&A[i + 7][j]);
			wmm5 = _mm256_mul_ps(wmmalpha, wmm0i7);
			wmm7 = _mm256_mul_ps(wmm3, wmm5);
			wmm8 = _mm256_mul_ps(wmm1i7, wmm4);
			wmm9 = _mm256_add_ps(wmm7, wmm8);
			wmm6i7 = _mm256_add_ps(wmm2, wmm9);

			_mm256_store_ps(&A[i][j], wmm6);
			_mm256_store_ps(&A[i + 1][j], wmm6i1);
			_mm256_store_ps(&A[i + 2][j], wmm6i2);
			_mm256_store_ps(&A[i + 3][j], wmm6i3);
			_mm256_store_ps(&A[i + 4][j], wmm6i4);
			_mm256_store_ps(&A[i + 5][j], wmm6i5);
			_mm256_store_ps(&A[i + 6][j], wmm6i6);
			_mm256_store_ps(&A[i + 7][j], wmm6i7);


			// A[i][j] += alpha * u1[i] * v1[j] + u2[i] * v2[j];
		}
	}
	

	
	__m256 xmm0, xmm1, xmm2, xmm3, xmm3i8, xmm3i16, xmm3i24, xmm3i32, xmm3i40, xmm3i48, xmm3i56, xmmbeta;

	xmmbeta = _mm256_set1_ps(beta); // load 8 copies of beta

	for (j = 0; j < N; j++) {
		xmm0 = _mm256_set1_ps(y[j]); // load 8 copies of y[]
		for (i = 0; i < N; i+=64) {
			xmm1 = _mm256_load_ps(&x[i]); // load 8 elements of x[]
			xmm2 = _mm256_load_ps(&A[j][i]); // load 8 elements of A[j][]

			xmm3 = _mm256_mul_ps(xmm2, xmm0); // A[][] * y[j]
			xmm3 = _mm256_add_ps(xmm3, xmmbeta); // (A[][] * y[j]) + beta
			xmm3 = _mm256_add_ps(xmm3, xmm1); // ((A[][] * y[j]) + beta) + x[]

			// i + 8
			xmm1 = _mm256_load_ps(&x[i + 8]);
			xmm2 = _mm256_load_ps(&A[j][i + 8]);
			xmm3i8 = _mm256_mul_ps(xmm2, xmm0);
			xmm3i8 = _mm256_add_ps(xmm3i8, xmmbeta);
			xmm3i8 = _mm256_add_ps(xmm3i8, xmm1);

			// i + 16
			xmm1 = _mm256_load_ps(&x[i + 16]);
			xmm2 = _mm256_load_ps(&A[j][i + 16]);
			xmm3i16 = _mm256_mul_ps(xmm2, xmm0);
			xmm3i16 = _mm256_add_ps(xmm3i16, xmmbeta);
			xmm3i16 = _mm256_add_ps(xmm3i16, xmm1);

			// i + 24
			xmm1 = _mm256_load_ps(&x[i + 24]);
			xmm2 = _mm256_load_ps(&A[j][i + 24]);
			xmm3i24 = _mm256_mul_ps(xmm2, xmm0);
			xmm3i24 = _mm256_add_ps(xmm3i24, xmmbeta);
			xmm3i24 = _mm256_add_ps(xmm3i24, xmm1);

			// i + 32
			xmm1 = _mm256_load_ps(&x[i + 32]);
			xmm2 = _mm256_load_ps(&A[j][i + 32]);
			xmm3i32 = _mm256_mul_ps(xmm2, xmm0);
			xmm3i32 = _mm256_add_ps(xmm3i32, xmmbeta);
			xmm3i32 = _mm256_add_ps(xmm3i32, xmm1);

			// i + 40
			xmm1 = _mm256_load_ps(&x[i + 40]);
			xmm2 = _mm256_load_ps(&A[j][i + 40]);
			xmm3i40 = _mm256_mul_ps(xmm2, xmm0);
			xmm3i40 = _mm256_add_ps(xmm3i40, xmmbeta);
			xmm3i40 = _mm256_add_ps(xmm3i40, xmm1);

			// i + 48
			xmm1 = _mm256_load_ps(&x[i + 48]);
			xmm2 = _mm256_load_ps(&A[j][i + 48]);
			xmm3i48 = _mm256_mul_ps(xmm2, xmm0);
			xmm3i48 = _mm256_add_ps(xmm3i48, xmmbeta);
			xmm3i48 = _mm256_add_ps(xmm3i48, xmm1);

			// i + 56
			xmm1 = _mm256_load_ps(&x[i + 56]);
			xmm2 = _mm256_load_ps(&A[j][i + 56]);
			xmm3i56 = _mm256_mul_ps(xmm2, xmm0);
			xmm3i56 = _mm256_add_ps(xmm3i56, xmmbeta);
			xmm3i56 = _mm256_add_ps(xmm3i56, xmm1);


			_mm256_store_ps(&x[i], xmm3); // store xmm3 into x[]
			_mm256_store_ps(&x[i + 8], xmm3i8);
			_mm256_store_ps(&x[i + 16], xmm3i16);
			_mm256_store_ps(&x[i + 24], xmm3i24);
			_mm256_store_ps(&x[i + 32], xmm3i32);
			_mm256_store_ps(&x[i + 40], xmm3i40);
			_mm256_store_ps(&x[i + 48], xmm3i48);
			_mm256_store_ps(&x[i + 56], xmm3i56);


		}
	}
		


	// Need to add loop for extra iterations (e.g., if N = 10 need to account for the 2 extra iterations)
	__m256 ymm0, ymm1, ymm2, ymm3;
	ymm0 = _mm256_set1_ps(3.22f); // set num values

	__m256 zmm0, zmm0i1, zmm0i2, zmm0i3, zmm0i4, zmm0i5, zmm0i6, zmm0i7, zmm1, zmm2, zmm3, zmm3i1, zmm3i2, zmm3i3, zmm3i4, zmm3i5, zmm3i6, zmm3i7, zmmalpha, zmmbeta;
	__m128 total;

	zmmalpha = _mm256_set1_ps(alpha); // 8 copies of alpha
	zmmbeta = _mm256_set1_ps(beta); //8 copies of beta

	for (i = 0; i < N; i+=8) {
		

		ymm3 = _mm256_load_ps(&z[i]); // load 8 elements of z[]
		ymm1 = _mm256_load_ps(&x[i]); // load 8 elements of x[]
		_mm256_fmadd_ps(ymm0, ymm3, ymm0); // multiply ymm0 and ymm3, store results into ymm0
		ymm2 = _mm256_add_ps(ymm1, ymm0); // add ymm1 and ymm0, store results into ymm2
		_mm256_store_ps(&x[i], ymm2); // store values of ymm2 into x[]

		// i + 1
		ymm3 = _mm256_load_ps(&z[i + 1]);
		ymm1 = _mm256_load_ps(&x[i + 1]); 
		_mm256_fmadd_ps(ymm0, ymm3, ymm0); 
		ymm2 = _mm256_add_ps(ymm1, ymm0);
		_mm256_store_ps(&x[i + 1], ymm2);

		// i + 2
		ymm3 = _mm256_load_ps(&z[i + 2]);
		ymm1 = _mm256_load_ps(&x[i + 2]);
		_mm256_fmadd_ps(ymm0, ymm3, ymm0);
		ymm2 = _mm256_add_ps(ymm1, ymm0);
		_mm256_store_ps(&x[i + 2], ymm2);

		// i + 3
		ymm3 = _mm256_load_ps(&z[i + 3]);
		ymm1 = _mm256_load_ps(&x[i + 3]);
		_mm256_fmadd_ps(ymm0, ymm3, ymm0);
		ymm2 = _mm256_add_ps(ymm1, ymm0);
		_mm256_store_ps(&x[i + 3], ymm2);

		// i + 4
		ymm3 = _mm256_load_ps(&z[i + 4]);
		ymm1 = _mm256_load_ps(&x[i + 4]);
		_mm256_fmadd_ps(ymm0, ymm3, ymm0);
		ymm2 = _mm256_add_ps(ymm1, ymm0);
		_mm256_store_ps(&x[i + 4], ymm2);

		// i + 5
		ymm3 = _mm256_load_ps(&z[i + 5]);
		ymm1 = _mm256_load_ps(&x[i + 5]);
		_mm256_fmadd_ps(ymm0, ymm3, ymm0);
		ymm2 = _mm256_add_ps(ymm1, ymm0);
		_mm256_store_ps(&x[i + 5], ymm2);

		// i + 6
		ymm3 = _mm256_load_ps(&z[i + 6]);
		ymm1 = _mm256_load_ps(&x[i + 6]);
		_mm256_fmadd_ps(ymm0, ymm3, ymm0);
		ymm2 = _mm256_add_ps(ymm1, ymm0);
		_mm256_store_ps(&x[i + 6], ymm2);

		// i + 7
		ymm3 = _mm256_load_ps(&z[i + 7]);
		ymm1 = _mm256_load_ps(&x[i + 7]);
		_mm256_fmadd_ps(ymm0, ymm3, ymm0);
		ymm2 = _mm256_add_ps(ymm1, ymm0);
		_mm256_store_ps(&x[i + 7], ymm2);

		zmm0 = _mm256_set1_ps(w[i]); // 8 copies of w[i]
		zmm3 = _mm256_setzero_ps();

		zmm0i1 = _mm256_set1_ps(w[i + 1]); 
		zmm3i1 = _mm256_setzero_ps();

		zmm0i2 = _mm256_set1_ps(w[i + 2]);
		zmm3i2 = _mm256_setzero_ps();

		zmm0i3 = _mm256_set1_ps(w[i + 3]);
		zmm3i3 = _mm256_setzero_ps();

		zmm0i4 = _mm256_set1_ps(w[i + 4]);
		zmm3i4 = _mm256_setzero_ps();

		zmm0i5 = _mm256_set1_ps(w[i + 5]);
		zmm3i5 = _mm256_setzero_ps();

		zmm0i6 = _mm256_set1_ps(w[i + 6]);
		zmm3i6 = _mm256_setzero_ps();

		zmm0i7 = _mm256_set1_ps(w[i + 7]);
		zmm3i7 = _mm256_setzero_ps();

		for (j = 0; j < N; j += 8) {
			zmm1 = _mm256_load_ps(&A[i][j]);
			zmm2 = _mm256_load_ps(&x[j]);
			zmm1 = _mm256_mul_ps(zmm1, zmm2); // A[][] * x[]
			zmm1 = _mm256_mul_ps(zmm1, zmmalpha); // (A[][] * x[]) * alpha
			zmm1 = _mm256_add_ps(zmm1, zmmbeta); // ((A[][] * x[]) * alpha) + beta
			zmm3 = _mm256_add_ps(zmm3, zmm1);

			// i + 1
			zmm1 = _mm256_load_ps(&A[i + 1][j]);
			zmm1 = _mm256_mul_ps(zmm1, zmm2);
			zmm1 = _mm256_mul_ps(zmm1, zmmalpha);
			zmm1 = _mm256_add_ps(zmm1, zmmbeta); 
			zmm3i1 = _mm256_add_ps(zmm3i1, zmm1);

			// i + 2
			zmm1 = _mm256_load_ps(&A[i + 2][j]);
			zmm1 = _mm256_mul_ps(zmm1, zmm2);
			zmm1 = _mm256_mul_ps(zmm1, zmmalpha);
			zmm1 = _mm256_add_ps(zmm1, zmmbeta);
			zmm3i2 = _mm256_add_ps(zmm3i2, zmm1);

			// i + 3
			zmm1 = _mm256_load_ps(&A[i + 3][j]);
			zmm1 = _mm256_mul_ps(zmm1, zmm2);
			zmm1 = _mm256_mul_ps(zmm1, zmmalpha);
			zmm1 = _mm256_add_ps(zmm1, zmmbeta);
			zmm3i3 = _mm256_add_ps(zmm3i3, zmm1);

			// i + 4
			zmm1 = _mm256_load_ps(&A[i + 4][j]);
			zmm1 = _mm256_mul_ps(zmm1, zmm2);
			zmm1 = _mm256_mul_ps(zmm1, zmmalpha);
			zmm1 = _mm256_add_ps(zmm1, zmmbeta);
			zmm3i4 = _mm256_add_ps(zmm3i4, zmm1);

			// i + 5
			zmm1 = _mm256_load_ps(&A[i + 5][j]);
			zmm1 = _mm256_mul_ps(zmm1, zmm2);
			zmm1 = _mm256_mul_ps(zmm1, zmmalpha);
			zmm1 = _mm256_add_ps(zmm1, zmmbeta);
			zmm3i5 = _mm256_add_ps(zmm3i5, zmm1);

			// i + 6
			zmm1 = _mm256_load_ps(&A[i + 6][j]);
			zmm1 = _mm256_mul_ps(zmm1, zmm2);
			zmm1 = _mm256_mul_ps(zmm1, zmmalpha);
			zmm1 = _mm256_add_ps(zmm1, zmmbeta);
			zmm3i6 = _mm256_add_ps(zmm3i6, zmm1);

			// i + 7
			zmm1 = _mm256_load_ps(&A[i + 7][j]);
			zmm1 = _mm256_mul_ps(zmm1, zmm2);
			zmm1 = _mm256_mul_ps(zmm1, zmmalpha);
			zmm1 = _mm256_add_ps(zmm1, zmmbeta);
			zmm3i7 = _mm256_add_ps(zmm3i7, zmm1);
		}

		zmm0 = _mm256_permute2f128_ps(zmm3, zmm3, 1);
		zmm3 = _mm256_add_ps(zmm3, zmm0);
		zmm3 = _mm256_hadd_ps(zmm3, zmm3); // pack zmm1 into one value for total
		zmm3 = _mm256_hadd_ps(zmm3, zmm3);
		total = _mm256_extractf128_ps(zmm3, 0); // extract packed zmm1
		_mm_store_ss(&w[i], total); // store total into w[i]

		// i + 1
		zmm0i1 = _mm256_permute2f128_ps(zmm3i1, zmm3i1, 1);
		zmm3i1 = _mm256_add_ps(zmm3i1, zmm0i1);
		zmm3i1 = _mm256_hadd_ps(zmm3i1, zmm3i1); 
		zmm3i1 = _mm256_hadd_ps(zmm3i1, zmm3i1);
		total = _mm256_extractf128_ps(zmm3i1, 0);
		_mm_store_ss(&w[i + 1], total); 

		// i + 2
		zmm0i2 = _mm256_permute2f128_ps(zmm3i2, zmm3i2, 1);
		zmm3i2 = _mm256_add_ps(zmm3i2, zmm0i2);
		zmm3i2 = _mm256_hadd_ps(zmm3i2, zmm3i2);
		zmm3i2 = _mm256_hadd_ps(zmm3i2, zmm3i2);
		total = _mm256_extractf128_ps(zmm3i2, 0);
		_mm_store_ss(&w[i + 2], total);

		// i + 3
		zmm0i3 = _mm256_permute2f128_ps(zmm3i3, zmm3i3, 1);
		zmm3i3 = _mm256_add_ps(zmm3i3, zmm0i3);
		zmm3i3 = _mm256_hadd_ps(zmm3i3, zmm3i3);
		zmm3i3 = _mm256_hadd_ps(zmm3i3, zmm3i3);
		total = _mm256_extractf128_ps(zmm3i3, 0);
		_mm_store_ss(&w[i + 3], total);

		// i + 4
		zmm0i4 = _mm256_permute2f128_ps(zmm3i4, zmm3i4, 1);
		zmm3i4 = _mm256_add_ps(zmm3i4, zmm0i4);
		zmm3i4 = _mm256_hadd_ps(zmm3i4, zmm3i4);
		zmm3i4 = _mm256_hadd_ps(zmm3i4, zmm3i4);
		total = _mm256_extractf128_ps(zmm3i4, 0);
		_mm_store_ss(&w[i + 4], total);

		// i + 5
		zmm0i5 = _mm256_permute2f128_ps(zmm3i5, zmm3i5, 1);
		zmm3i5 = _mm256_add_ps(zmm3i5, zmm0i5);
		zmm3i5 = _mm256_hadd_ps(zmm3i5, zmm3i5);
		zmm3i5 = _mm256_hadd_ps(zmm3i5, zmm3i5);
		total = _mm256_extractf128_ps(zmm3i5, 0);
		_mm_store_ss(&w[i + 5], total);

		// i + 6
		zmm0i6 = _mm256_permute2f128_ps(zmm3i6, zmm3i6, 1);
		zmm3i6 = _mm256_add_ps(zmm3i6, zmm0i6);
		zmm3i6 = _mm256_hadd_ps(zmm3i6, zmm3i6);
		zmm3i6 = _mm256_hadd_ps(zmm3i6, zmm3i6);
		total = _mm256_extractf128_ps(zmm3i6, 0);
		_mm_store_ss(&w[i + 6], total);

		// i + 7
		zmm0i7 = _mm256_permute2f128_ps(zmm3i7, zmm3i7, 1);
		zmm3i7 = _mm256_add_ps(zmm3i7, zmm0i7);
		zmm3i7 = _mm256_hadd_ps(zmm3i7, zmm3i7);
		zmm3i7 = _mm256_hadd_ps(zmm3i7, zmm3i7);
		total = _mm256_extractf128_ps(zmm3i7, 0);
		_mm_store_ss(&w[i + 7], total);
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


