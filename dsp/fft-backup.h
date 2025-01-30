#ifndef SIGNALSMITH_FFT_V5
#define SIGNALSMITH_FFT_V5

#include <vector>
#include <complex>
#include "complex_ops.h"
typedef std::complex<float> Complex;
//Due idee: ottimizzazione di questo codice oppure impiego al suo posto della libreria FFTW
Complex complexMul(bool conjugateSecond, Complex a, Complex b) {
	Complex res = conjugateSecond ? Complex{
		b.real()*a.real() + b.imag()*a.imag(),
		b.real()*a.imag() - b.imag()*a.real()
	} : Complex{
		a.real()*b.real() - a.imag()*b.imag(),
		a.real()*b.imag() + a.imag()*b.real()
	};
	return res;
}
Complex complexAddI(bool flipped, Complex a, Complex b) {
	return flipped ? Complex{
		a.real() + b.imag(),
		a.imag() - b.real()
	} : Complex{
		a.real() - b.imag(),
		a.imag() + b.real()
	};
}
	
class FFT {
	size_t _size;	
	enum class StepType {
		generic, step2, step3, step4
	};
	struct Step {
		StepType type;
		size_t factor;
		size_t startIndex;
		size_t innerRepeats;
		size_t outerRepeats;
		size_t twiddleIndex;
	};
	std::vector<Complex> workingVector;
	std::vector<size_t> factors;
	std::vector<Step> plan;
	std::vector<Complex> twiddleVector;
	
	struct PermutationPair {size_t from, to;};
	std::vector<PermutationPair> permutation;
	
	void addPlanSteps(size_t factorIndex, size_t start, size_t length, size_t repeats) {
		if (factorIndex >= factors.size()) 
			return;
		
		size_t factor = factors[factorIndex];
		if (factorIndex + 1 < factors.size()) {
			if (factors[factorIndex] == 2 && factors[factorIndex + 1] == 2) {
				++factorIndex;
				factor = 4;
			}
		}
		size_t subLength = length/factor;
		Step mainStep{StepType::generic, factor, start, subLength, repeats, twiddleVector.size()};
		mainStep.type = StepType::step4;
		// Twiddles
		bool foundStep = false;
		for (const Step &existingStep : plan) {
			if (existingStep.factor == mainStep.factor && existingStep.innerRepeats == mainStep.innerRepeats) {
				foundStep = true;
				mainStep.twiddleIndex = existingStep.twiddleIndex;
				break;
			}
		}
		if (!foundStep) {
			for (size_t i = 0; i < subLength; ++i) {
				for (size_t f = 0; f < factor; ++f) {
					float phase = 2*M_PI*i*f/length;
					Complex twiddle;
					twiddle.real(std::cos(phase));
					twiddle.imag(-std::sin(phase));
					twiddleVector.push_back(twiddle);
				}
			}
		}

		for (size_t i = 0; i < factor; ++i) {
			addPlanSteps(factorIndex + 1, start + i*subLength, subLength, 1);
		}
		plan.push_back(mainStep);
	}
	void setPlan() {
		factors.resize(0);
		size_t size = _size, factor = 2;
		while (size > 1) {
			factors.push_back(factor);
			size /= factor;
		}
		plan.resize(0);
		twiddleVector.resize(0);
		addPlanSteps(0, 0, _size, 1);
		
		permutation.resize(0);
		permutation.push_back(PermutationPair{0, 0});
		size_t indexLow = 0, indexHigh = factors.size();
		size_t inputStepLow = _size, outputStepLow = 1;
		size_t inputStepHigh = 1, outputStepHigh = _size;
		while (outputStepLow*inputStepHigh < _size) {
			size_t f, inputStep, outputStep;
			if (outputStepLow <= inputStepHigh) {
				f = factors[indexLow++];
				inputStep = (inputStepLow /= f);
				outputStep = outputStepLow;
				outputStepLow *= f;
			} else {
				f = factors[--indexHigh];
				inputStep = inputStepHigh;
				inputStepHigh *= f;
				outputStep = (outputStepHigh /= f);
			}
			size_t oldSize = permutation.size();
			for (size_t i = 1; i < f; ++i) {
				for (size_t j = 0; j < oldSize; ++j) {
					PermutationPair pair = permutation[j];
					pair.from += i*inputStep;
					pair.to += i*outputStep;
					permutation.push_back(pair);
				}
			}
		}
	}

	static bool validSize(size_t size) {
		constexpr static bool filter[32] = {
			1, 1, 1, 1, 1, 0, 1, 0, 1, 1, // 0-9
			0, 0, 1, 0, 0, 0, 1, 0, 1, 0, // 10-19
			0, 0, 0, 0, 1, 0, 0, 0, 0, 0, // 20-29
			0, 0
		};
		return filter[size];
	}
	public:
		static size_t fastSizeAbove(size_t size) {
			size_t power2 = 1;
			while (size >= 32) {
				size = (size - 1)/2 + 1;
				power2 *= 2;
			}
			while (size < 32 && !validSize(size)) {
				++size;
			}
			return power2*size;
		}

		FFT(size_t size, int fastDirection=0) : _size(0) {
			this->setSize(size);
		}

		size_t setSize(size_t size) {
			if (size != _size) {
				_size = size;
				workingVector.resize(size);
				setPlan();
			}
			return _size;
		}
		size_t setFastSizeAbove(size_t size) {
			return setSize(fastSizeAbove(size));
		}
		const size_t & size() const {
			return _size;
		}

	void run(bool inverse, Complex *input, Complex *data) {
		for(PermutationPair pair : permutation)
		{
			data[pair.from]=input[pair.to];
		}
		for (const Step &step : plan) {
			const size_t stride = step.innerRepeats;
			const Complex *origTwiddles = twiddleVector.data() + step.twiddleIndex;
			Complex* origData = data + step.startIndex;
			for (size_t outerRepeat = 0; outerRepeat < step.outerRepeats; ++outerRepeat) {
				const Complex* twiddles = origTwiddles;
				for (Complex* data = origData; data < origData + stride; ++data) {
					Complex A = data[0];
					Complex C = complexMul(inverse, data[stride], twiddles[2]);
					Complex B = complexMul(inverse, data[stride*2], twiddles[1]);
					Complex D = complexMul(inverse, data[stride*3], twiddles[3]);
					Complex sumAC = A + C, sumBD = B + D;
					Complex diffAC = A - C, diffBD = B - D;
					data[0] = sumAC + sumBD;
					data[stride] = complexAddI(!inverse,diffAC, diffBD);
					data[stride*2] = sumAC - sumBD;
					data[stride*3] = complexAddI(inverse, diffAC, diffBD);
					twiddles += 4;
				}
				origData += 4*stride;
			}
		}
	}
};

class RealFFT {
	static const int optionFlags=0;
	std::vector<Complex> complexBuffer1, complexBuffer2;
	std::vector<Complex> twiddlesMinusI;
	FFT complexFft;
	public:
		static size_t fastSizeAbove(size_t size) {
			return FFT::fastSizeAbove((size + 1)/2)*2;
		}

		RealFFT(size_t size=0, int fastDirection=0) : complexFft(0) {
			this->setSize(std::max<size_t>(size, 2));
		}

		size_t setSize(size_t size) {
			complexBuffer1.resize(size/2);
			complexBuffer2.resize(size/2);

			size_t hhSize = size/4 + 1;
			twiddlesMinusI.resize(hhSize);
			for (size_t i = 0; i < hhSize; ++i) {
				float rotPhase = -2*M_PI*i/size;
				twiddlesMinusI[i].real(std::sin(rotPhase));
				twiddlesMinusI[i].imag(-std::cos(rotPhase));
			}			
			return complexFft.setSize(size/2);
		}
		size_t setFastSizeAbove(size_t size) {
			return setSize(fastSizeAbove(size));
		}
		size_t size() const {
			return complexFft.size()*2;
		}

		void fft(float *input, Complex* output) {
			size_t hSize = complexFft.size();
			for (size_t i = 0; i < hSize; ++i) {
				complexBuffer1[i] = {input[2*i], input[2*i + 1]};
			}
			
			complexFft.run(false, complexBuffer1.data(), complexBuffer2.data());
			output[0] = {
				complexBuffer2[0].real() + complexBuffer2[0].imag(),
				complexBuffer2[0].real() - complexBuffer2[0].imag()
			};
			for (size_t i = 1; i <= hSize/2; ++i) {
				size_t conjI = hSize - i;
				
				Complex odd = (complexBuffer2[i] + conj(complexBuffer2[conjI]))*0.5f;
				Complex evenI = (complexBuffer2[i] - conj(complexBuffer2[conjI]))*0.5f;
				Complex evenRotMinusI = complexMul(false, evenI, twiddlesMinusI[i]);

				output[i] = odd + evenRotMinusI;
				output[conjI] = conj(odd - evenRotMinusI);
			}
		}
		void ifft(Complex *input, float *&output) {
			size_t hSize = complexFft.size();
			complexBuffer1[0] = {
				input[0].real() + input[0].imag(),
				input[0].real() - input[0].imag()
			};
			for (size_t i = 1; i <= hSize/2; ++i) {
				size_t conjI = hSize - i;
				Complex v = input[i], v2 = input[conjI];

				Complex odd = v + conj(v2);
				Complex evenRotMinusI = v - conj(v2);
				Complex evenI = complexMul(true, evenRotMinusI, twiddlesMinusI[i]);
				
				complexBuffer1[i] = odd + evenI;
				complexBuffer1[conjI] = conj(odd - evenI);
			}
			
			complexFft.run(true, complexBuffer1.data(), complexBuffer2.data());
			
			for (size_t i = 0; i < hSize; ++i) {
				Complex v = complexBuffer2[i];
				output[2*i] = v.real();
				output[2*i + 1] = v.imag();
			}
		}
};
#endif