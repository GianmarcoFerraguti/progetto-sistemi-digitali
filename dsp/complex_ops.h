#ifndef COMPLEX_OPS
#define COMPLEX_OPS
#include <immintrin.h>
#define REAL 0
#define IMAG 1
typedef __bfloat16 fftw_complex[2];
void complexCopy(fftw_complex dest, fftw_complex src)
{
	dest[REAL] = src[REAL];
	dest[IMAG] = src[IMAG];
}
void complexCopy(fftw_complex dest, __bfloat16 srcReal, __bfloat16 srcImag)
{
    dest[REAL] = srcReal;
    dest[IMAG] = srcImag;
}
void complexSum(fftw_complex res, fftw_complex a, fftw_complex b, bool diff)
{
	if(diff)
	{
		b[REAL] *= -1;
		b[IMAG] *= -1;
	}
	res[REAL] = a[REAL] + b[REAL];
	res[IMAG] = a[IMAG] + b[IMAG];
}
void complexMulmine(bool conjugateSecond, fftw_complex res, fftw_complex a, fftw_complex b) {
	if(conjugateSecond)
		complexCopy(res, b[REAL]*a[REAL] + b[IMAG]*a[IMAG], b[REAL]*a[IMAG] - b[IMAG]*a[REAL]);
	else
		complexCopy(res, a[REAL]*b[REAL] - a[IMAG]*b[IMAG], a[REAL]*b[IMAG] + a[IMAG]*b[REAL]);
}
#endif