#ifndef SIGNALSMITH_EXAMPLE_SHIFT_STRETCH_H
#define SIGNALSMITH_EXAMPLE_SHIFT_STRETCH_H
#define M_PI 3.14159265358979323846264338327950288
#include "dsp/delay.h"
#include "dsp/fft.h"
#include "dsp/windows.h"
#include <complex>

int size[9];
typedef enum {
	BLOCK_BUFFERS=0, WINDOW=1, FFT_BUFFER, CHANNEL_SPECTRA, ENERGY, SMOOTHED_ENERGY, PREV_SPECTRA, NEW_SPECTRA, PREV_OUTPUT_ROTATIONS
} Array_Names;
typedef std::complex<double> Complex;//using Complex = std::complex<Sample>;
typedef Complex complex;
RealFFT mrfft;
class SpectralCutStretch {
public:
	int bandCount = 0;
	double scalingFactor = 1;
	int channels = 0, blockSamples = 0;
	int intervalSamples = 0, intervalCounter = 0;
	double invTimeFactor = 1;
	
	MultiBuffer inputHistory, summedOutput;
	int maxSurplusInputSamples = 0;
	double surplusInputSamples = 0;
	int prevInputIndex = 0;

	double freqFactor = 1;
	double* fftBuffer;
	double *energy, *smoothedEnergy;
	double* blockBuffers, *window;
	Complex* channelSpectra;
	Complex* newSpectra, *prevSpectra;
	Complex* prevOutputRotations;

	SpectralCutStretch() {}

	void configure(int channels, int blockSamples, int intervalSamples, double zeroPadding=2, int maxExtraInput=0) {
		//Sono tutti e 4 interi a 32 bit: si potrebbero raggruppare in un registro esteso da 128 bit
		this->channels = channels;
		this->blockSamples = blockSamples;
		this->intervalSamples = intervalSamples;
		this->maxSurplusInputSamples = maxExtraInput;

		inputHistory.resize(channels, blockSamples + maxExtraInput,0);
		summedOutput.resize(channels, blockSamples,0);
		size[BLOCK_BUFFERS]=blockSamples*channels;
		size[WINDOW]=blockSamples;
		blockBuffers=(double*)malloc(sizeof(double)*size[BLOCK_BUFFERS]);
		window=(double*)malloc(sizeof(double)*size[WINDOW]);
		Kaiser kaiser = Kaiser::withBandwidth(blockSamples*1.0/intervalSamples, true);
		kaiser.fill(window, blockSamples);
		// Makes it add up nicely to 1 when applied twice
		forcePerfectReconstruction(window, blockSamples, intervalSamples);
		intervalCounter = 0;
		mrfft.setFastSizeAbove(blockSamples*zeroPadding);
		bandCount = mrfft.size()/2;
		scalingFactor = 1.0/mrfft.size(); // the FFT round-trip scales things up, so we scale down again
		size[FFT_BUFFER]=mrfft.size();
		size[CHANNEL_SPECTRA]=bandCount*channels;
		size[ENERGY]=size[SMOOTHED_ENERGY]=size[NEW_SPECTRA]=size[PREV_SPECTRA]=size[PREV_OUTPUT_ROTATIONS]=bandCount;
		//Dove si possono deallocare? Quando non servono più o definendo un metodo finalize per dealloare tutto alla fine?
		fftBuffer=(double*)malloc(sizeof(double)*size[FFT_BUFFER]);
		channelSpectra=(Complex*)malloc(sizeof(Complex)*size[CHANNEL_SPECTRA]);
		energy=(double*)malloc(sizeof(double)*bandCount);
		smoothedEnergy=(double*)malloc(sizeof(double)*bandCount);
		newSpectra=(Complex*)malloc(sizeof(Complex)*(bandCount*channels));
		prevSpectra=(Complex*)malloc(sizeof(Complex)*(bandCount*channels));
		prevOutputRotations=(Complex*)malloc(sizeof(Complex)*bandCount);

		//Possibile parallelizzazione in SIMD: due celle per ciascun elemento, una per la parte reale e una per la parte immaginaria
		for (int b = 0; b < bandCount; ++b) {
			double phase = ((b+0.5f)/mrfft.size())*(-intervalSamples)*(-2*M_PI);
			prevOutputRotations[b] = {std::cos(phase), std::sin(phase)};
		}
	}
	
	void process(double** inputs, int inputSamples, double **outputs, int outputSamples) {
		int inputFilledTo = 0;
		for (int o = 0; o < outputSamples; ++o) {
			if (++intervalCounter >= intervalSamples) {
				intervalCounter = 0;
				// Fill the block from the input
				int inputStart = int(std::round(o*invTimeFactor - surplusInputSamples - blockSamples));
				// For safety: don't go past the end of the block, or too far in the past
				inputStart = std::max(std::min(inputStart, inputSamples - blockSamples), -maxSurplusInputSamples - blockSamples);
				//Si potrebbe parallelizzare, ma prima occorre togliere l'OOP da delay.h e definire tutto in termini di tipi primitivi
				for (int c = 0; c < channels; ++c) {
					// Make sure we have enough input history
					double* input = inputs[c];
					Buffer::View history = inputHistory[c];
					for (int i = inputFilledTo; i < inputStart + blockSamples; ++i) {
						history[i] = input[i];
					}
					// Fill the block from history
					double *blockBuffer = &blockBuffers[c*blockSamples];
					for (int i = 0; i < blockSamples; ++i) {
						blockBuffer[i] = history[inputStart + i]*window[i];
					}
				}
				
				processBlock(inputStart - prevInputIndex);
				prevInputIndex = inputStart;
				
				// Add the block to the summed output
				//Potenzialmente parallelizzabile, a patto di sistemare delay.h come già detto
				for (int c = 0; c < channels; ++c) {
					double *blockBuffer = &blockBuffers[c*blockSamples];
					Buffer::View output = summedOutput[c];
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
			double* input = inputs[c];
			Buffer::View history = inputHistory[c];
			for (int i = inputFilledTo; i < inputSamples; ++i) {
				history[i] = input[i];
			}
		}
		inputHistory += inputSamples;
		prevInputIndex -= inputSamples;
		surplusInputSamples += inputSamples - outputSamples*invTimeFactor;
	}

	void processBlock(int inputIntervalSamples) {
		for (int c = 0; c < this->channels; ++c) {
			double *block = &blockBuffers[c*blockSamples];
			Complex *spectrum = &channelSpectra[c*bandCount];
			for (int i = 0; i < this->blockSamples; ++i) {
				fftBuffer[i] = block[i];
			}
			// Zero-padding
			for (int i = this->blockSamples; i < size[FFT_BUFFER]; ++i) {
				fftBuffer[i] = 0;
			}
			mrfft.fft(fftBuffer, spectrum);
		}

		processSpectrum(inputIntervalSamples);

		for (int c = 0; c < this->channels; ++c) {
			double *block = &blockBuffers[c*blockSamples];
			Complex *spectrum = &channelSpectra[c*bandCount];
			mrfft.ifft(spectrum, fftBuffer);
			for (int i = 0; i < this->blockSamples; ++i) {
				block[i] = fftBuffer[i]*scalingFactor;
			}
		}
	}

	void processSpectrum(int inputIntervalSamples) {
		for (int b = 0; b < bandCount; ++b) {
			double e = 0;
			for (int c = 0; c < this->channels; ++c) {
				Complex bin = (channelSpectra+(c*bandCount))[b];
				e += std::norm(bin); // magnitude squared
			}
			energy[b] = smoothedEnergy[b] = e;
		}
		
		double smoothingFactor = 0.25; // Really this should depend on your overlap-ratio and stuff, but this whole thing's a bit approximate
		double smooth = energy[0];
		for (int b = 1; b < bandCount; ++b) { // smooth upwards
			smooth += (smoothedEnergy[b] - smooth)*smoothingFactor;
			smoothedEnergy[b] = smooth;
		}
		for (int b = bandCount - 1; b >= 0; --b) { // smooth downwards
			smooth += (smoothedEnergy[b] - smooth)*smoothingFactor;
			smoothedEnergy[b] = smooth;
		}
		
		int binIndex = 0;
		int prevSegmentStart = 0, prevSegmentEnd = 0;
		while (binIndex < bandCount) {
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
				while (segmentEnd < bandCount && energy[segmentEnd] > smoothedEnergy[segmentEnd]*0.25f) {
					++segmentEnd;
				}
				prevSegmentEnd = binIndex = segmentEnd;
			} else {
				++binIndex;
			}
		}
		// Extend final band to the end, and copy it in
		copySegmentToNew(prevSegmentStart, bandCount);
		
		// Copy the new spectrum across
		//Potenzialmente parallelizzabile in SIMD
		for (int c = 0; c < this->channels; ++c) {
			Complex *spectrum = &channelSpectra[c*bandCount];
			Complex *newSpectrum = &newSpectra[c*bandCount];
			Complex *prevSpectrum = &prevSpectra[c*bandCount];
			for (int b = 0; b < bandCount; ++b) {
				spectrum[b] = newSpectrum[b];
				newSpectrum[b] = 0;
				prevSpectrum[b] = spectrum[b]*prevOutputRotations[b];
			}
		}
	}
	
	// Copy a segment of the spectrum to the output spectrum, shifted in frequency
	void copySegmentToNew(int segmentStart, int segmentEnd) {
		// find centre of the segment by energy-weighted average
		double binTotal = 0, energyTotal = 0;
		for (int b = segmentStart; b < segmentEnd; ++b) {
			binTotal += b*energy[b];
			energyTotal += energy[b];
		}
		double binAverage = binTotal/(energyTotal + 1e-100);
		double centreFreq = (binAverage+0.5f)/mrfft.size();//this->bandToFreq(binAverage);
		double newCentreFreq = centreFreq*freqFactor;
		int binOffset = std::round(newCentreFreq*mrfft.size() - 0.5f - binAverage); //freqToBand(newCentreFreq)

		Complex phaseShift = 1;
		Complex phaseShiftSum = 0;
		for (int c = 0; c < this->channels; ++c) {
			Complex *spectrum = &channelSpectra[c*bandCount];
			Complex *prevSpectrum = &prevSpectra[c*bandCount];
			for (int b = segmentStart; b < segmentEnd; ++b) {
				int newB = b + binOffset;
				if (newB > 0 && newB < bandCount) {
					phaseShiftSum += prevSpectrum[newB]*std::conj(spectrum[b]);
				}
			}
		}
		double norm = std::norm(phaseShiftSum);
		if (norm > 0) {
			phaseShift = phaseShiftSum/std::sqrt(norm);
		}
		for (int c = 0; c < this->channels; ++c) {
			Complex *spectrum = &channelSpectra[c*bandCount];
			Complex *newSpectrum = &newSpectra[c*bandCount];
			for (int b = segmentStart; b < segmentEnd; ++b) {
				int newB = b + binOffset;
				if (newB > 0 && newB < bandCount) {
					newSpectrum[newB] += spectrum[b]*phaseShift;
				}
			}
		}
	}
};

#endif
