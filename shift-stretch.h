#ifndef SIGNALSMITH_EXAMPLE_SHIFT_STRETCH_H
#define SIGNALSMITH_EXAMPLE_SHIFT_STRETCH_H
#define M_PI 3.14159265358979323846264338327950288
#include "dsp/delay.h"
#include "dsp/fft.h"
#include "dsp/complex_ops.h"
#include "dsp/windows.h"
class SpectralCutStretch {
public:
	int size[9];
	typedef enum {
		BLOCK_BUFFERS=0, WINDOW=1, FFT_BUFFER, CHANNEL_SPECTRA, ENERGY, SMOOTHED_ENERGY, PREV_SPECTRA, NEW_SPECTRA, PREV_OUTPUT_ROTATIONS
	} Array_Names;
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
	fftw_complex* channelSpectra;
	fftw_complex* newSpectra, *prevSpectra;
	fftw_complex* prevOutputRotations;
	RealFFT mrfft;

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
		Kaiser kaiser = Kaiser::withBandwidth(blockSamples*1.0/intervalSamples);
		kaiser.fill(window, blockSamples);
		// Makes it add up nicely to 1 when applied twice
		intervalCounter = 0;
		mrfft.setFastSizeAbove(blockSamples*zeroPadding);
		size[FFT_BUFFER]=mrfft.size();
		bandCount = mrfft.size()/2;
		scalingFactor = 1.0/mrfft.size(); // the FFT round-trip scales things up, so we scale down again
		size[CHANNEL_SPECTRA]=bandCount*channels;
		size[ENERGY]=size[SMOOTHED_ENERGY]=size[NEW_SPECTRA]=size[PREV_SPECTRA]=size[PREV_OUTPUT_ROTATIONS]=bandCount;
		fftBuffer=(double*)malloc(sizeof(double)*size[FFT_BUFFER]);
		channelSpectra=(fftw_complex*)malloc(sizeof(fftw_complex)*(bandCount*channels));
		energy=(double*)malloc(sizeof(double)*bandCount);
		smoothedEnergy=(double*)malloc(sizeof(double)*bandCount);
		newSpectra=(fftw_complex*)malloc(sizeof(fftw_complex)*(bandCount*channels));
		prevSpectra=(fftw_complex*)malloc(sizeof(fftw_complex)*(bandCount*channels));
		prevOutputRotations=(fftw_complex*)malloc(sizeof(fftw_complex)*bandCount);

		//Possibile parallelizzazione in SIMD: due celle per ciascun elemento, una per la parte reale e una per la parte immaginaria
		for (int b = 0; b < bandCount; ++b) {
			double phase = ((b+0.5f)/size[FFT_BUFFER])*(-intervalSamples)*(-2*M_PI);
			prevOutputRotations[b][REAL] = std::cos(phase);
			prevOutputRotations[b][IMAG] = std::sin(phase);
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
					View history(inputHistory.buffer, inputHistory.bufferIndex, inputHistory.bufferMask, c*inputHistory.stride);
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
					View output(summedOutput.buffer, summedOutput.bufferIndex, summedOutput.bufferMask, c*summedOutput.stride);
					for (int i = 0; i < blockSamples; ++i) {
						output[i] += blockBuffer[i]*window[i];
					}
				}
			}
			//Potenzialmente parallelizzabile in
			for (int c = 0; c < channels; ++c) {
				View view(summedOutput.buffer, summedOutput.bufferIndex, summedOutput.bufferMask,c * summedOutput.stride);
				outputs[c][o] = view[0];
				view[0] = 0;
			}
			summedOutput.bufferIndex++;
		}
		
		// Copy in remaining input
		for (int c = 0; c < channels; ++c) {
			double* input = inputs[c];
			View history(inputHistory.buffer, inputHistory.bufferIndex, inputHistory.bufferMask, c*inputHistory.stride);
			for (int i = inputFilledTo; i < inputSamples; ++i) {
				history[i] = input[i];
			}
		}
		inputHistory.bufferIndex += inputSamples; //buffer.bufferIndex += i;
		prevInputIndex -= inputSamples;
		surplusInputSamples += inputSamples - outputSamples*invTimeFactor;
	}

	void processBlock(int inputIntervalSamples) {
		Complex* oSpectrum = (Complex*)malloc(sizeof(Complex)*size[FFT_BUFFER]);
		for (int c = 0; c < this->channels; ++c) {
			double *block = &blockBuffers[c*blockSamples];
			fftw_complex *spectrum = &channelSpectra[c*bandCount];
			for (int i = 0; i < this->blockSamples; ++i) {
				fftBuffer[i] = block[i];
			}
			// Zero-padding
			for (int i = this->blockSamples; i < size[FFT_BUFFER]; ++i) {
				fftBuffer[i] = 0;
			}
			mrfft.fft(fftBuffer, oSpectrum);
			for(int i = 0; i < size[FFT_BUFFER]/2; i++)
			{
				complexCopy(spectrum[i], oSpectrum[i].real(), oSpectrum[i].imag());
			}
		}

		processSpectrum(inputIntervalSamples);

		for (int c = 0; c < this->channels; ++c) {
			double *block = &blockBuffers[c*blockSamples];
			fftw_complex *spectrum = &channelSpectra[c*bandCount];
			for(int i = 0; i < size[FFT_BUFFER]; i++)
			{
				oSpectrum[i] = std::complex<double>{spectrum[i][REAL], spectrum[i][IMAG]};
			}
			mrfft.ifft(oSpectrum, fftBuffer);
			for (int i = 0; i < this->blockSamples; ++i) {
				block[i] = fftBuffer[i]*scalingFactor;
			}
		}
	}

	void processSpectrum(int inputIntervalSamples) {
		for (int b = 0; b < bandCount; ++b) {
			double e = 0;
			for (int c = 0; c < this->channels; ++c) {
				fftw_complex bin;
				complexCopy(bin, (channelSpectra+(c*bandCount))[b]);
				e += bin[REAL] * bin[REAL] + bin[IMAG] * bin[IMAG]; //std::norm(bin); // magnitude squared
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
		fftw_complex zero, product;
		zero[REAL] = 0;
		zero[IMAG] = 0;
		for (int c = 0; c < this->channels; ++c) {
			fftw_complex *spectrum = &channelSpectra[c*bandCount];
			fftw_complex *newSpectrum = &newSpectra[c*bandCount];
			fftw_complex *prevSpectrum = &prevSpectra[c*bandCount];
			for (int b = 0; b < bandCount; ++b) {
				complexCopy(spectrum[b], newSpectrum[b]);
				complexCopy(newSpectrum[b], zero);
				complexMulmine(false, product, spectrum[b], prevOutputRotations[b]);
				complexCopy(prevSpectrum[b], product);
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
		double centreFreq = (binAverage+0.5f)/size[FFT_BUFFER];//this->bandToFreq(binAverage);
		double newCentreFreq = centreFreq*freqFactor;
		int binOffset = std::round(newCentreFreq*size[FFT_BUFFER] - 0.5f - binAverage); //freqToBand(newCentreFreq)

		fftw_complex phaseShift;
		phaseShift[REAL] = 1;
		phaseShift[IMAG] = 0;
		fftw_complex phaseShiftSum;
		phaseShiftSum[REAL] = 0;
		phaseShiftSum[IMAG] = 0;
		fftw_complex product;
		//Probabile possibilità di riscrittura in SIMD
		for (int c = 0; c < this->channels; ++c) {
			fftw_complex *spectrum = &channelSpectra[c*bandCount];
			fftw_complex *prevSpectrum = &prevSpectra[c*bandCount];
			for (int b = segmentStart; b < segmentEnd; ++b) {
				int newB = b + binOffset;
				if (newB > 0 && newB < bandCount) {
					complexMulmine(true, product, prevSpectrum[newB], spectrum[b]);
					complexSum(phaseShiftSum, phaseShiftSum, product, false);
				}
			}
		}
		double norm = phaseShiftSum[REAL] * phaseShiftSum[REAL] + phaseShiftSum[IMAG] * phaseShiftSum[IMAG]; //std::norm(phaseShiftSum);
		if (norm > 0) {
			phaseShift[REAL] = phaseShiftSum[REAL]/std::sqrt(norm);
			phaseShift[IMAG] = phaseShiftSum[IMAG]/std::sqrt(norm);
		}
		//Probabile possibilità di riscrittura in SIMD
		for (int c = 0; c < this->channels; ++c) {
			fftw_complex *spectrum = &channelSpectra[c*bandCount];
			fftw_complex *newSpectrum = &newSpectra[c*bandCount];
			for (int b = segmentStart; b < segmentEnd; ++b) {
				int newB = b + binOffset;
				if (newB > 0 && newB < bandCount) {
					complexMulmine(false, product, spectrum[b], phaseShift);
					complexSum(newSpectrum[newB], newSpectrum[newB], product, false);
				}
			}
		}
	}
	void finalize()
	{
		free(fftBuffer);
		free(blockBuffers);
		free(window);
		free(channelSpectra);
		free(energy);
		free(smoothedEnergy);
		free(newSpectra);
		free(prevSpectra);
		free(prevOutputRotations);
	}
};

#endif