#ifndef COMPLEX_OPS
#define COMPLEX_OPS
#include <immintrin.h>
#include <fftw3.h>
#define REAL 0
#define IMAG 1
typedef float fftwf_complex[2];
void complexCopy(fftwf_complex dest, fftwf_complex src)
{
	dest[REAL] = src[REAL];
	dest[IMAG] = src[IMAG];
}
void complexCopy(fftwf_complex dest, float srcReal, float srcImag)
{
    dest[REAL] = srcReal;
    dest[IMAG] = srcImag;
}
void complexSum(fftwf_complex res, fftwf_complex a, fftwf_complex b, bool diff)
{
	if(diff)
	{
		b[REAL] *= -1;
		b[IMAG] *= -1;
	}
	res[REAL] = a[REAL] + b[REAL];
	res[IMAG] = a[IMAG] + b[IMAG];
}
void complexMulmine(bool conjugateSecond, fftwf_complex res, fftwf_complex a, fftwf_complex b) {
	if(conjugateSecond)
		complexCopy(res, b[REAL]*a[REAL] + b[IMAG]*a[IMAG], b[REAL]*a[IMAG] - b[IMAG]*a[REAL]);
	else
		complexCopy(res, a[REAL]*b[REAL] - a[IMAG]*b[IMAG], a[REAL]*b[IMAG] + a[IMAG]*b[REAL]);
}
#endif