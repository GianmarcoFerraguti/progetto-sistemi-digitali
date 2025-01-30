#ifndef SIGNALSMITH_EXAMPLE_SHIFT_STRETCH_H
#define SIGNALSMITH_EXAMPLE_SHIFT_STRETCH_H
#define M_PI 3.14159265358979323846264338327950288F
#include "dsp/delay.h"
#include "dsp/fft-backup.h"
#include "dsp/complex_ops.h"
#include "dsp/windows.h"
#include "util/simd_consts.h"
class SpectralCutStretch {
public:
	int size[9];
	typedef enum {
		BLOCK_BUFFERS=0, WINDOW=1, FFT_BUFFER, CHANNEL_SPECTRA, ENERGY, SMOOTHED_ENERGY, PREV_SPECTRA, NEW_SPECTRA, PREV_OUTPUT_ROTATIONS
	} Array_Names;
	int bandCount = 0;
	float scalingFactor = 1;
	int channels = 0, blockSamples = 0;
	int intervalSamples = 0, intervalCounter = 0;
	float invTimeFactor = 1;
	
	MultiBuffer inputHistory, summedOutput;
	int maxSurplusInputSamples = 0;
	float surplusInputSamples = 0;
	int prevInputIndex = 0;

	float freqFactor = 1;
	float* fftBuffer;
	float *energy, *smoothedEnergy;
	float* blockBuffers, *window;
	fftwf_complex* channelSpectra;
	fftwf_complex* newSpectra, *prevSpectra;
	fftwf_complex* prevOutputRotations;
	RealFFT mrfft;

	SpectralCutStretch() {}

	void configure(int channels, int blockSamples, int intervalSamples, float zeroPadding=2, int maxExtraInput=0) {
		this->channels = channels;
		this->blockSamples = blockSamples;
		this->intervalSamples = intervalSamples;
		this->maxSurplusInputSamples = maxExtraInput;

		inputHistory.resize(channels, blockSamples + maxExtraInput,0);
		summedOutput.resize(channels, blockSamples,0);
		size[BLOCK_BUFFERS]=blockSamples*channels;
		size[WINDOW]=blockSamples;
		blockBuffers=(float*)_mm_malloc(sizeof(float)*size[BLOCK_BUFFERS], AVX_DATA_LANE);
		window=(float*)_mm_malloc(sizeof(float)*size[WINDOW], AVX_DATA_LANE);
		kaiserWithBandwidth(blockSamples*1.0/intervalSamples);
		kaiserFill(window, blockSamples);
		// Makes it add up nicely to 1 when applied twice
		intervalCounter = 0;
		mrfft.setFastSizeAbove(blockSamples*zeroPadding);
		size[FFT_BUFFER]=mrfft.size();
		bandCount = mrfft.size()/2;
		scalingFactor = 1.0/mrfft.size(); // the FFT round-trip scales things up, so we scale down again
		size[CHANNEL_SPECTRA]=bandCount*channels;
		size[ENERGY]=size[SMOOTHED_ENERGY]=size[NEW_SPECTRA]=size[PREV_SPECTRA]=size[PREV_OUTPUT_ROTATIONS]=bandCount;
		fftBuffer=(float*)_mm_malloc(sizeof(float)*size[FFT_BUFFER], AVX_DATA_LANE);
		channelSpectra=(fftwf_complex*)_mm_malloc(sizeof(fftwf_complex)*(bandCount*channels), AVX_DATA_LANE);
		energy=(float*)_mm_malloc(sizeof(float)*bandCount, AVX_DATA_LANE);
		smoothedEnergy=(float*)_mm_malloc(sizeof(float)*bandCount, AVX_DATA_LANE);
		newSpectra=(fftwf_complex*)_mm_malloc(sizeof(fftwf_complex)*(bandCount*channels), AVX_DATA_LANE);
		prevSpectra=(fftwf_complex*)_mm_malloc(sizeof(fftwf_complex)*(bandCount*channels), AVX_DATA_LANE);
		prevOutputRotations=(fftwf_complex*)_mm_malloc(sizeof(fftwf_complex)*bandCount, AVX_DATA_LANE);

		//Possibile parallelizzazione in SIMD: due celle per ciascun elemento, una per la parte reale e una per la parte immaginaria
		for (int b = 0; b < bandCount; ++b) {
			float phase = ((b+0.5f)/size[FFT_BUFFER])*(-intervalSamples)*(-2*M_PI);
			complexCopy(prevOutputRotations[b], (float)(std::cos((float)phase)), (float)(std::sin((float)phase)));
		}
	}
	
	void process(float** inputs, int inputSamples, float **outputs, int outputSamples) {
		int inputFilledTo = 0;
		for (int o = 0; o < outputSamples; ++o) {
			if (++intervalCounter >= intervalSamples) {
				intervalCounter = 0;
				// Fill the block from the input
				int inputStart = int(std::round((o*invTimeFactor - surplusInputSamples - blockSamples)));
				// For safety: don't go past the end of the block, or too far in the past
				inputStart = std::max(std::min(inputStart, inputSamples - blockSamples), -maxSurplusInputSamples - blockSamples);
				//Si potrebbe parallelizzare, ma prima occorre togliere l'OOP da delay.h e definire tutto in termini di tipi primitivi
				for (int c = 0; c < channels; ++c) {
					// Make sure we have enough input history
					float* input = inputs[c];
					View history(inputHistory.buffer, inputHistory.bufferIndex, inputHistory.bufferMask, c*inputHistory.stride);
					for (int i = inputFilledTo; i < inputStart + blockSamples; ++i) {
						history[i] = input[i];
					}
					// Fill the block from history
					float *blockBuffer = &blockBuffers[c*blockSamples];
					for (int i = 0; i < blockSamples; ++i) {
						blockBuffer[i] = history[inputStart + i]*window[i];
					}
				}
				
				processBlock(inputStart - prevInputIndex); //Metodo da parallelizzare
				prevInputIndex = inputStart;
				
				// Add the block to the summed output
				//Potenzialmente parallelizzabile, a patto di sistemare delay.h come già detto
				for (int c = 0; c < channels; ++c) {
					float *blockBuffer = &blockBuffers[c*blockSamples];
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
			float* input = inputs[c];
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
		Complex* oSpectrum = (Complex*)_mm_malloc(sizeof(Complex)*size[FFT_BUFFER], AVX_DATA_LANE);
		__m512 XMM_BLOCK, XMM_FFT_BUFFER, XMM_SCALE_FACTOR;
		__m512* pBlock;
		XMM_SCALE_FACTOR = _mm512_set1_ps(scalingFactor);
		for (int c = 0; c < this->channels; ++c) {
			float *block = &blockBuffers[c*blockSamples];
			fftwf_complex *spectrum = &channelSpectra[c*bandCount];
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
			float *block = &blockBuffers[c*blockSamples];
			fftwf_complex *spectrum = &channelSpectra[c*bandCount];
			for(int i = 0; i < size[FFT_BUFFER]; i++)
			{
				oSpectrum[i] = std::complex<float>{spectrum[i][REAL], spectrum[i][IMAG]};
			}
			mrfft.ifft(oSpectrum, fftBuffer);
			for (int i = 0; i < this->blockSamples; ++i) {
				block[i] = fftBuffer[i]*scalingFactor;
			}
		}
	}

	void processSpectrum(int inputIntervalSamples) {
		for (int b = 0; b < bandCount; ++b) {
			float e = 0;
			for (int c = 0; c < this->channels; ++c) {
				fftwf_complex bin;
				complexCopy(bin, (channelSpectra+(c*bandCount))[b]);
				e += bin[REAL] * bin[REAL] + bin[IMAG] * bin[IMAG]; //std::norm(bin); // magnitude squared
			}
			energy[b] = smoothedEnergy[b] = e;
		}
		
		float smoothingFactor = 0.25F; // Really this should depend on your overlap-ratio and stuff, but this whole thing's a bit approximate
		float smooth = energy[0];
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
					while (segmentStart > 0 && energy[segmentStart] > smoothedEnergy[segmentStart]*0.25F) {
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
		fftwf_complex zero, product;
		zero[REAL] = 0;
		zero[IMAG] = 0;
		for (int c = 0; c < this->channels; ++c) {
			fftwf_complex *spectrum = &channelSpectra[c*bandCount];
			fftwf_complex *newSpectrum = &newSpectra[c*bandCount];
			fftwf_complex *prevSpectrum = &prevSpectra[c*bandCount];
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
		float binTotal = 0, energyTotal = 0;
		for (int b = segmentStart; b < segmentEnd; ++b) {
			binTotal += b*energy[b];
			energyTotal += energy[b];
		}
		float binAverage = binTotal/(energyTotal + 1e-100);
		float centreFreq = (binAverage+0.5f)/size[FFT_BUFFER];//this->bandToFreq(binAverage);
		float newCentreFreq = centreFreq*freqFactor;
		int binOffset = std::round(newCentreFreq*size[FFT_BUFFER] - 0.5f - binAverage); //freqToBand(newCentreFreq)

		fftwf_complex phaseShift;
		phaseShift[REAL] = 1;
		phaseShift[IMAG] = 0;
		fftwf_complex phaseShiftSum;
		phaseShiftSum[REAL] = 0;
		phaseShiftSum[IMAG] = 0;
		fftwf_complex product;
		//Probabile possibilità di riscrittura in SIMD
		for (int c = 0; c < this->channels; ++c) {
			fftwf_complex *spectrum = &channelSpectra[c*bandCount];
			fftwf_complex *prevSpectrum = &prevSpectra[c*bandCount];
			for (int b = segmentStart; b < segmentEnd; ++b) {
				int newB = b + binOffset;
				if (newB > 0 && newB < bandCount) {
					complexMulmine(true, product, prevSpectrum[newB], spectrum[b]);
					complexSum(phaseShiftSum, phaseShiftSum, product, false);
				}
			}
		}
		float norm = phaseShiftSum[REAL] * phaseShiftSum[REAL] + phaseShiftSum[IMAG] * phaseShiftSum[IMAG]; //std::norm(phaseShiftSum);
		if (norm > 0) {
			phaseShift[REAL] = phaseShiftSum[REAL]/std::sqrt(norm);
			phaseShift[IMAG] = phaseShiftSum[IMAG]/std::sqrt(norm);
		}
		//Probabile possibilità di riscrittura in SIMD
		for (int c = 0; c < this->channels; ++c) {
			fftwf_complex *spectrum = &channelSpectra[c*bandCount];
			fftwf_complex *newSpectrum = &newSpectra[c*bandCount];
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
		_mm_free(fftBuffer);
		_mm_free(blockBuffers);
		_mm_free(window);
		_mm_free(channelSpectra);
		_mm_free(energy);
		_mm_free(smoothedEnergy);
		_mm_free(newSpectra);
		_mm_free(prevSpectra);
		_mm_free(prevOutputRotations);
	}
};

#endif