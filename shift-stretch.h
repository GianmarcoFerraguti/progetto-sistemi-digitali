#ifndef SIGNALSMITH_EXAMPLE_SHIFT_STRETCH_H
#define SIGNALSMITH_EXAMPLE_SHIFT_STRETCH_H

#include "dsp/delay.h"
#include "dsp/fft.h"
#include <complex>
#include <vector>

#ifndef LOG_EXPR
#	include <iostream>
#	define LOG_EXPR(expr) std::cout << #expr << " = " << (expr) << std::endl;
#endif

class OverlapAddStretch {
public:
	using Sample = double;

	OverlapAddStretch(bool isSpectral=false) : isSpectral(isSpectral) {}
	
	void configure(int channels, int blockSamples, int intervalSamples, int maxExtraInput=0) {
		this->channels = channels;
		this->blockSamples = blockSamples;
		this->intervalSamples = intervalSamples;
		this->maxSurplusInputSamples = maxExtraInput;

		inputHistory.resize(channels, blockSamples + maxExtraInput);
		summedOutput.resize(channels, blockSamples);
		blockBuffers.resize(blockSamples*channels);
		window.resize(blockSamples);
		if (isSpectral) {
			// Kaiser's a good window for spectral stuff, but not so great for time-domain
			auto kaiser = signalsmith::windows::Kaiser::withBandwidth(blockSamples*1.0/intervalSamples, true);
			kaiser.fill(window, blockSamples);
		} else {
			for (int i = 0; i < blockSamples; ++i) {
				double r = (i + 0.5)/blockSamples;
				window[i] = std::sin(r*M_PI); // sine window, becomes Hann when applied twice
			}
		}
		// Makes it add up nicely to 1 when applied twice
		signalsmith::windows::forcePerfectReconstruction(window, blockSamples, intervalSamples);
		
		intervalCounter = 0;
	}
	
	void reset() {
		inputHistory.reset();
		summedOutput.reset();
		intervalCounter = 0;
	}
	
	void setRate(double rate) {
		invTimeFactor = rate;
	}
	void setTimeFactor(double timeFactor) {
		invTimeFactor = 1/timeFactor;
	}
	/// How many input samples do we need to get this much output?
	int samplesForOutput(int outputSamples) const {
		double inputSamples = outputSamples*invTimeFactor - surplusInputSamples;
		return int(std::ceil(inputSamples));
	}
	
	void process(const Sample * const *inputs, int inputSamples, Sample **outputs, int outputSamples) {
		int inputFilledTo = 0;
		for (int o = 0; o < outputSamples; ++o) {
			if (++intervalCounter >= intervalSamples) {
				intervalCounter = 0;
				// Fill the block from the input
				int inputStart = int(std::round(o*invTimeFactor - surplusInputSamples - blockSamples));
				// For safety: don't go past the end of the block, or too far in the past
				inputStart = std::max(std::min(inputStart, inputSamples - blockSamples), -maxSurplusInputSamples - blockSamples);
				for (int c = 0; c < channels; ++c) {
					// Make sure we have enough input history
					auto input = inputs[c];
					auto history = inputHistory[c];
					for (int i = inputFilledTo; i < inputStart + blockSamples; ++i) {
						history[i] = input[i];
					}
					// Fill the block from history
					Sample *blockBuffer = channelBlock(c);
					for (int i = 0; i < blockSamples; ++i) {
						blockBuffer[i] = history[inputStart + i]*window[i];
					}
				}
				
				processBlock(inputStart - prevInputIndex);
				prevInputIndex = inputStart;
				
				// Add the block to the summed output
				for (int c = 0; c < channels; ++c) {
					Sample *blockBuffer = channelBlock(c);
					auto output = summedOutput[c];
					for (int i = 0; i < blockSamples; ++i) {
						output[i] += blockBuffer[i]*window[i];
					}
				}
			}
			for (int c = 0; c < channels; ++c) {
				outputs[c][o] = summedOutput[c][0];
				summedOutput[c][0] = 0;
			}
			++summedOutput;
		}
		
		// Copy in remaining input
		for (int c = 0; c < channels; ++c) {
			auto input = inputs[c];
			auto history = inputHistory[c];
			for (int i = inputFilledTo; i < inputSamples; ++i) {
				history[i] = input[i];
			}
		}
		inputHistory += inputSamples;
		prevInputIndex -= inputSamples;
		surplusInputSamples += inputSamples - outputSamples*invTimeFactor;
	}
	
	int inputLatency() const {
		return blockSamples/2;
	}
	int outputLatency() const {
		return blockSamples - inputLatency();
	}

protected:
	int channels = 0, blockSamples = 0;
	int intervalSamples = 0, intervalCounter = 0;
	double invTimeFactor = 1;
	
	Sample * channelBlock(int channel) {
		return blockBuffers.data() + channel*blockSamples;
	}
	
	virtual void processBlock(int inputIntervalSamples) {
		// Alter the blocks (for each channel) if we want to
		(void)inputIntervalSamples;
	}

	void scheduleNextBlock(int interval) {
		intervalCounter = intervalSamples - interval;
	}
private:
	bool isSpectral;

	// Multi-channel circular buffers
	signalsmith::delay::MultiBuffer<Sample> inputHistory, summedOutput;
	std::vector<Sample> blockBuffers, window;

	// Unused input samples, which may be fractional
	int maxSurplusInputSamples = 0;
	double surplusInputSamples = 0;
	int prevInputIndex = 0;
};

class SpectralStretch : public OverlapAddStretch {
public:
	using Complex = std::complex<Sample>;

	SpectralStretch(bool kaiserWindow=true) : OverlapAddStretch(kaiserWindow) {}
	
	void configure(int channels, int blockSamples, int intervalSamples, double zeroPadding=1, int maxExtraInput=0) {
		OverlapAddStretch::configure(channels, blockSamples, intervalSamples, maxExtraInput);
		
		mrfft.setFastSizeAbove(blockSamples*zeroPadding);
		fftBuffer.resize(mrfft.size());
		bandCount = mrfft.size()/2;
		scalingFactor = 1.0/mrfft.size(); // the FFT round-trip scales things up, so we scale down again
		channelSpectra.resize(bandCount*channels);
	}
protected:
	virtual void processSpectrum(int inputIntervalSamples) {
		// Edit the spectrums using `channelSpectrum()`, `bands()` and `bandToFreq()`/`freqToBand()`
		(void)inputIntervalSamples;
	}

	Complex * channelSpectrum(int channel) {
		return channelSpectra.data() + channel*bandCount;
	}

	int bands() const {
		return bandCount;
	}
	int fftSize() const {
		return int(mrfft.size());
	}
	Sample bandToFreq(Sample band) const {
		return (band + 0.5f)/mrfft.size();
	}
	Sample freqToBand(Sample freq) const {
		return freq*mrfft.size() - 0.5f;
	}

	void timeShiftPhases(Sample shiftSamples, Complex *output) const {
		for (int b = 0; b < bandCount; ++b) {
			Sample phase = bandToFreq(b)*shiftSamples*(-2*M_PI);
			output[b] = {std::cos(phase), std::sin(phase)};
		}
	}

	void processBlock(int inputIntervalSamples) override final {
		for (int c = 0; c < this->channels; ++c) {
			Sample *block = this->channelBlock(c);
			Complex *spectrum = channelSpectrum(c);
			for (int i = 0; i < this->blockSamples; ++i) {
				fftBuffer[i] = block[i];
			}
			// Zero-padding
			for (int i = this->blockSamples; i < int(fftBuffer.size()); ++i) {
				fftBuffer[i] = 0;
			}
			mrfft.fft(fftBuffer, spectrum);
		}

		processSpectrum(inputIntervalSamples);

		for (int c = 0; c < this->channels; ++c) {
			Sample *block = this->channelBlock(c);
			Complex *spectrum = channelSpectrum(c);
			mrfft.ifft(spectrum, fftBuffer);
			for (int i = 0; i < this->blockSamples; ++i) {
				block[i] = fftBuffer[i]*scalingFactor;
			}
		}
	}

	static Complex generateComplex(Sample energy, Complex complexPhase) {
		Sample complexPhaseNorm = std::norm(complexPhase);
		if (complexPhaseNorm > 0) {
			return complexPhase*std::sqrt(energy/complexPhaseNorm);
		} else {
			Sample phase = Sample(2*M_PI)*rand()/RAND_MAX;
			Complex complexPhase = {std::cos(phase), std::sin(phase)};
			return std::sqrt(energy)*complexPhase;
		}
	}

private:
	signalsmith::fft::ModifiedRealFFT<Sample> mrfft{1};
	int bandCount = 0;
	Sample scalingFactor = 1;
	std::vector<Sample> fftBuffer;
	std::vector<Complex> channelSpectra;
};

class SpectralCutStretch : public SpectralStretch {
public:
	SpectralCutStretch(bool fixedPhase) : fixedPhase(fixedPhase) {}

	void configure(int channels, int blockSamples, int intervalSamples, double zeroPadding=2, int maxExtraInput=0) {
		SpectralStretch::configure(channels, blockSamples, intervalSamples, zeroPadding, maxExtraInput);
		
		energy.resize(this->bands());
		smoothedEnergy.resize(this->bands());
		newSpectra.resize(this->bands()*channels);
		prevSpectra.resize(this->bands()*channels);
		
		prevOutputRotations.resize(bands());
		timeShiftPhases(-intervalSamples, prevOutputRotations.data());
	}
	
	void reset() {
		prevSpectra.assign(prevSpectra.size(), 0);
	}

	void setFreqFactor(double factor) {
		freqFactor = factor;
	}
protected:
	virtual void processSpectrum(int) {
		for (int b = 0; b < this->bands(); ++b) {
			Sample e = 0;
			for (int c = 0; c < this->channels; ++c) {
				Complex bin = this->channelSpectrum(c)[b];
				e += std::norm(bin); // magnitude squared
			}
			energy[b] = smoothedEnergy[b] = e;
		}
		
		Sample smoothingFactor = 0.25; // Really this should depend on your overlap-ratio and stuff, but this whole thing's a bit approximate
		Sample smooth = energy[0];
		for (int b = 1; b < this->bands(); ++b) { // smooth upwards
			smooth += (smoothedEnergy[b] - smooth)*smoothingFactor;
			smoothedEnergy[b] = smooth;
		}
		for (int b = this->bands() - 1; b >= 0; --b) { // smooth downwards
			smooth += (smoothedEnergy[b] - smooth)*smoothingFactor;
			smoothedEnergy[b] = smooth;
		}
		
		int binIndex = 0;
		int prevSegmentStart = 0, prevSegmentEnd = 0;
		while (binIndex < this->bands()) {
			if (energy[binIndex] > smoothedEnergy[binIndex]) {
				if (prevSegmentEnd > 0) { // if it's not the first segment
					// backtrack until it's 6dB below the smoothed energy
					int segmentStart = binIndex;
					while (segmentStart > 0 && energy[segmentStart] > smoothedEnergy[segmentStart]*0.25f) {
						--segmentStart;
					}
				
					// extend this segment back and the previous one forwards
					int midPoint = (segmentStart + prevSegmentEnd)/2;
					// copy the previous segment across
					copySegmentToNew(prevSegmentStart, midPoint);
					prevSegmentStart = midPoint;
				}
				
				// and extend forward until it's 6dB below the smoothed energy
				int segmentEnd = binIndex + 1;
				while (segmentEnd < this->bands() && energy[segmentEnd] > smoothedEnergy[segmentEnd]*0.25f) {
					++segmentEnd;
				}
				prevSegmentEnd = binIndex = segmentEnd;
			} else {
				++binIndex;
			}
		}
		// Extend final band to the end, and copy it in
		copySegmentToNew(prevSegmentStart, this->bands());
		
		// Copy the new spectrum across
		for (int c = 0; c < this->channels; ++c) {
			Complex *spectrum = this->channelSpectrum(c);
			Complex *newSpectrum = newChannelSpectrum(c);
			Complex *prevSpectrum = prevChannelSpectrum(c);
			for (int b = 0; b < this->bands(); ++b) {
				spectrum[b] = newSpectrum[b];
				newSpectrum[b] = 0;
				prevSpectrum[b] = spectrum[b]*prevOutputRotations[b];
			}
		}
	}
private:
	bool fixedPhase;
	double freqFactor = 1;
	std::vector<Sample> energy, smoothedEnergy;
	std::vector<Complex> newSpectra, prevSpectra;
	Complex * newChannelSpectrum(int channel) {
		return newSpectra.data() + channel*this->bands();
	}
	Complex * prevChannelSpectrum(int channel) {
		return prevSpectra.data() + channel*this->bands();
	}
	std::vector<Complex> prevOutputRotations;
	
	// Copy a segment of the spectrum to the output spectrum, shifted in frequency
	void copySegmentToNew(int segmentStart, int segmentEnd) {
		// find centre of the segment by energy-weighted average
		double binTotal = 0, energyTotal = 0;
		for (int b = segmentStart; b < segmentEnd; ++b) {
			binTotal += b*energy[b];
			energyTotal += energy[b];
		}
		double binAverage = binTotal/(energyTotal + 1e-100);
		Sample centreFreq = this->bandToFreq(binAverage);
		Sample newCentreFreq = centreFreq*freqFactor;
		int binOffset = std::round(this->freqToBand(newCentreFreq) - binAverage);

		Complex phaseShift = 1;
		if (!fixedPhase) {
			Complex phaseShiftSum = 0;
			for (int c = 0; c < this->channels; ++c) {
				Complex *spectrum = this->channelSpectrum(c);
				Complex *prevSpectrum = prevChannelSpectrum(c);
				for (int b = segmentStart; b < segmentEnd; ++b) {
					int newB = b + binOffset;
					if (newB > 0 && newB < this->bands()) {
						phaseShiftSum += prevSpectrum[newB]*std::conj(spectrum[b]);
					}
				}
			}
			Sample norm = std::norm(phaseShiftSum);
			if (norm > 0) {
				phaseShift = phaseShiftSum/std::sqrt(norm);
			}
		}

		for (int c = 0; c < this->channels; ++c) {
			Complex *spectrum = this->channelSpectrum(c);
			Complex *newSpectrum = newChannelSpectrum(c);
			for (int b = segmentStart; b < segmentEnd; ++b) {
				int newB = b + binOffset;
				if (newB > 0 && newB < this->bands()) {
					newSpectrum[newB] += spectrum[b]*phaseShift;
				}
			}
		}
	}
};

#endif // include guard
