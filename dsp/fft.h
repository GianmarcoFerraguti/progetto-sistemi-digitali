#ifndef SIGNALSMITH_FFT_V5
#define SIGNALSMITH_FFT_V5
#include <fftw3.h>
class RealFFT {
	fftwf_plan p, q;
	size_t _size;
	public:
		size_t setFastSizeAbove(size_t size, float* in, fftwf_complex *out) {
			_size = size;
			p = fftwf_plan_dft_r2c_1d(size, in, out, FFTW_MEASURE);
			q = fftwf_plan_dft_c2r_1d(size, out, in, FFTW_MEASURE);
			return size;
		}
		size_t size(){
			return _size;
		}

		void fft(float *input, fftwf_complex* output) {
			fftwf_execute_dft_r2c(p, input, output);
		}
		void ifft(fftwf_complex *input, float *&output) {
			fftwf_execute_dft_c2r(q, input, output);
		}
};
#endif