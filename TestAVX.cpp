
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <memory.h>

#include <mmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>

void FloatMulAddAVX(float* mat, float* x, float* vec, float* y, int row, int col)
{
	int i, j;
	float* px;
	register __m256 xmm, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, zmm;

	for(i=0; i<row; i+=8) {
		zmm = _mm256_load_ps(vec);
		vec += 8;
		px = x;
		for(j=0; j<col-7; j+=8) {
			// load 8 cols
			xmm = _mm256_load_ps(px), px += 8;

			// load 8 rows
			ymm1 = _mm256_load_ps(mat);
			ymm2 = _mm256_load_ps(mat+8);
			ymm3 = _mm256_load_ps(mat+16);
			ymm4 = _mm256_load_ps(mat+24);
			ymm5 = _mm256_load_ps(mat+32);
			ymm6 = _mm256_load_ps(mat+40);
			ymm7 = _mm256_load_ps(mat+48);
			ymm8 = _mm256_load_ps(mat+56);
			mat += 64;

			// mul 8 rows
			ymm1 = _mm256_mul_ps(ymm1, xmm);
			ymm2 = _mm256_mul_ps(ymm2, xmm);
			ymm3 = _mm256_mul_ps(ymm3, xmm);
			ymm4 = _mm256_mul_ps(ymm4, xmm);
			ymm5 = _mm256_mul_ps(ymm5, xmm);
			ymm6 = _mm256_mul_ps(ymm6, xmm);
			ymm7 = _mm256_mul_ps(ymm7, xmm);
			ymm8 = _mm256_mul_ps(ymm8, xmm);
		
			// hadd 8 rows
			ymm1 = _mm256_hadd_ps(ymm1, ymm2); // a1+a2 a3+a4 b1+b2 b3+b4 a5+a6 a7+a8 b5+b6 b7+b8
			ymm2 = _mm256_hadd_ps(ymm3, ymm4); // c1+c2 c3+c4 d1+d2 d3+d4 c5+c6 c7+c8 d5+d6 d7+d8
			ymm3 = _mm256_hadd_ps(ymm5, ymm6); // p1+p2 p3+p4 q1+q2 q3+q4 p5+p6 p7+p8 q5+q6 q7+q8
			ymm4 = _mm256_hadd_ps(ymm7, ymm8); // r1+r2 r3+r4 s1+s2 s3+s4 r5+r6 r7+r8 s5+s6 s7+s8
			ymm1 = _mm256_hadd_ps(ymm1, ymm2); // a1~a4 b1~b4 c1~c4 d1~d4 a5~a8 b5~b8 c5~c8 d5~d8
			ymm2 = _mm256_hadd_ps(ymm3, ymm4); // p1~p4 q1~q4 r1~r4 s1~s4 p5~p8 q5~q8 r5~r8 s5~s8
			
			// pack 8 rows
			ymm3 = _mm256_blend_ps(ymm1, ymm2, 0b11110000);  // a1~a4 b1~b4 c1~c4 d1~d4  p5~p8 q5~q8 r5~r8 s5~s8
			ymm4 = _mm256_permute2f128_ps(ymm1, ymm2, 0x21); // a5~a8 b5~b8 c5~c8 d5~d8  p1~p4 q1~q4 r1~r4 s1~s4
			zmm = _mm256_add_ps(zmm, _mm256_add_ps(ymm3, ymm4));  // a1~a8 b1~b8 c1~c8 d1~d8  p1~p8 q1~q8 r1~r8 s1~s8
		}
		_mm256_store_ps(y, zmm);
		y += 8;
	}
}

void TestFloat(int dim, int count)
{
	int i, j, k;
	float *mat, *x, *pm;
	float *vec, *y;
	mat = (float*)_mm_malloc( sizeof(float) * dim * dim + dim * 3 * sizeof(float), 32);
	x = mat + dim * dim;
	y = x + dim;
	vec = (float*) (y + dim);

	//	init by random
	srand(314);
	for(i=0; i<dim; i++) {
		for(j=0; j<dim; j++) {
			mat[i*dim+j] = rand() % 2048 - 1024;
		}
		x[i] = rand() % 2048;
		vec[i] = rand() % 2048 - 1024;
	}

	//	matrix calculation
	time_t tt1 = clock();
	for(k=0; k<count; k++) {
		pm = mat;
		for(i=0; i<dim; i++) {
			y[i] = vec[i];
			for(j=0; j<dim; j++) {
				y[i] += pm[j] * x[j];
			}
			pm += dim;
		}
	}
	tt1 = clock() - tt1;

	time_t tt2 = clock();
	for(k=0; k<count; k++) {
		FloatMulAddAVX(mat, x, vec, y, dim, dim);
	}
	tt2 = clock() - tt2;

	float MA = 1.0f * 1.0E-9f * count * dim * dim;
	printf("Plain: %5.2f GIPS (%6.3f Sec)\t", MA * CLOCKS_PER_SEC / (float)tt1, (float)tt1/CLOCKS_PER_SEC);
	printf("AVX: %5.2f GIPS (%6.3f Sec)\n", MA * CLOCKS_PER_SEC / (float)tt2, (float)tt2/CLOCKS_PER_SEC);

	_mm_free( mat );
}

int main(int argc, char* argv[])
{
	int i, count = 10000;
	printf("\nTestFloat:\n");
	for(i=256; i<=1536; i+=256) {
		printf("%6d(%5.2f G MA)\t", i, count * 1.0e-9f * i * i);
		TestFloat(i, count);
	}
	printf("\nDone!\n\n");

	return 0;
}
