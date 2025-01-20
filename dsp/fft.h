#ifndef SIGNALSMITH_FFT_V5
#define SIGNALSMITH_FFT_V5

#include <vector>
#include <complex>
typedef std::complex<double> complex;

complex complexMul(bool conjugateSecond, const complex &a, const complex &b) {
	return conjugateSecond ? complex{
		b.real()*a.real() + b.imag()*a.imag(),
		b.real()*a.imag() - b.imag()*a.real()
	} : complex{
		a.real()*b.real() - a.imag()*b.imag(),
		a.real()*b.imag() + a.imag()*b.real()
	};
}
complex complexAddI(bool flipped, const complex &a, const complex &b) {
	return flipped ? complex{
		a.real() + b.imag(),
		a.imag() - b.real()
	} : complex{
		a.real() - b.imag(),
		a.imag() + b.real()
	};
}
	
class FFT {
	size_t _size;
	std::vector<complex> workingVector;
	
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
	std::vector<size_t> factors;
	std::vector<Step> plan;
	std::vector<complex> twiddleVector;
	
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
					double phase = 2*M_PI*i*f/length;
					complex twiddle = {(std::cos(phase)), (-std::sin(phase))};
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

	void run(bool inverse, complex *input, complex *data) {
		for(auto pair : permutation)
		{
			data[pair.from]=input[pair.to];
		}
		for (const Step &step : plan) {
			const size_t stride = step.innerRepeats;
			const complex *origTwiddles = twiddleVector.data() + step.twiddleIndex;
			complex* origData = data + step.startIndex;
			for (size_t outerRepeat = 0; outerRepeat < step.outerRepeats; ++outerRepeat) {
				const complex* twiddles = origTwiddles;
				for (complex* data = origData; data < origData + stride; ++data) {
					complex A = data[0];
					complex C = complexMul(inverse, data[stride], twiddles[2]);
					complex B = complexMul(inverse, data[stride*2], twiddles[1]);
					complex D = complexMul(inverse, data[stride*3], twiddles[3]);
					complex sumAC = A + C, sumBD = B + D;
					complex diffAC = A - C, diffBD = B - D;
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

struct FFTOptions {
	static constexpr int halfFreqShift = 1;
};

class RealFFT {
	static const int optionFlags=0;
	std::vector<complex> complexBuffer1, complexBuffer2;
	std::vector<complex> twiddlesMinusI;
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
				double rotPhase = -2*M_PI*i/size;
				twiddlesMinusI[i] = {std::sin(rotPhase), -std::cos(rotPhase)};
			}			
			return complexFft.setSize(size/2);
		}
		size_t setFastSizeAbove(size_t size) {
			return setSize(fastSizeAbove(size));
		}
		size_t size() const {
			return complexFft.size()*2;
		}

		void fft(double *&input, complex *&output) {
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
				
				complex odd = (complexBuffer2[i] + conj(complexBuffer2[conjI]))*0.5;
				complex evenI = (complexBuffer2[i] - conj(complexBuffer2[conjI]))*0.5;
				complex evenRotMinusI = complexMul(false, evenI, twiddlesMinusI[i]);

				output[i] = odd + evenRotMinusI;
				output[conjI] = conj(odd - evenRotMinusI);
			}
		}
		void ifft(complex *&input, double *&output) {
			size_t hSize = complexFft.size();
			complexBuffer1[0] = {
				input[0].real() + input[0].imag(),
				input[0].real() - input[0].imag()
			};
			for (size_t i = 1; i <= hSize/2; ++i) {
				size_t conjI = hSize - i;
				complex v = input[i], v2 = input[conjI];

				complex odd = v + conj(v2);
				complex evenRotMinusI = v - conj(v2);
				complex evenI = complexMul(true, evenRotMinusI, twiddlesMinusI[i]);
				
				complexBuffer1[i] = odd + evenI;
				complexBuffer1[conjI] = conj(odd - evenI);
			}
			
			complexFft.run(true, complexBuffer1.data(), complexBuffer2.data());
			
			for (size_t i = 0; i < hSize; ++i) {
				complex v = complexBuffer2[i];
				output[2*i] = v.real();
				output[2*i + 1] = v.imag();
			}
		}
};
#endif