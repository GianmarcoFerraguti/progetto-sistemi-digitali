#include "shift-stretch.h"
#include "util/wav.h"
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif // _WIN32
void pitchShifter(char* inputFile, char* outputFile, double semitones) {
	Wav inputWav, outputWav;
	double timeFactor = 1, freqFactor = 1;
	double blockMs = 80, overlapFactor = 4;
	long startTime, endTime;
	if (!inputWav.read(inputFile)) 
	{
		printf(inputWav.result.reason.c_str());
		exit(-1);
	}
	outputWav.channels = inputWav.channels;
	outputWav.sampleRate = inputWav.sampleRate;
	startTime=__rdtsc();
	freqFactor = pow(2,semitones/12);
	int blockSamples = int(blockMs*0.001*inputWav.sampleRate + 0.5);
	int intervalSamples = int(blockSamples/overlapFactor);
	SpectralCutStretch stretch; //Default constructor
	stretch.configure(inputWav.channels, blockSamples, intervalSamples);
	stretch.setTimeFactor(timeFactor);
	stretch.setFreqFactor(freqFactor);

	int channels = inputWav.channels;
	int blockSize = 256;
	
	std::vector<std::vector<double>> inputBuffers(channels), outputBuffers(channels);
	std::vector<double *> inputPointers(channels), outputPointers(channels);
	for (auto &b : outputBuffers) 
		b.resize(blockSize);
	
	outputWav.channels = inputWav.channels;
	int inputOffset = 0, outputOffset = 0;
	int inputLength = int(inputWav.length());
	int totalLatency = std::round(stretch.inputLatency()*timeFactor + stretch.outputLatency());
	int outputLength = inputWav.length()*timeFactor;
	while (outputOffset < outputLength + totalLatency*2) {
		int inputSamples = int(std::ceil(blockSize*stretch.invTimeFactor - stretch.surplusInputSamples));
		if (inputSamples > int(inputBuffers[0].size())) {
			for (auto &b : inputBuffers) 
				b.resize(inputSamples);
		}
		for (int c = 0; c < channels; ++c) {
			for (int i = 0; i < inputSamples; ++i) {
				if (inputOffset + i < inputLength) {
					inputBuffers[c][i] = inputWav[c][inputOffset + i];
				} else {
					inputBuffers[c][i] = 0;
				}
			}
			inputPointers[c] = inputBuffers[c].data();
			outputPointers[c] = outputBuffers[c].data();
		}
		
		stretch.process(inputPointers.data(), inputSamples, outputPointers.data(), blockSize);
		
		outputWav.samples.resize((outputOffset + blockSize)*channels);
		for (int c = 0; c < channels; ++c) {
			for (int i = 0; i < blockSize; ++i) {
				outputWav[c][outputOffset + i] = outputBuffers[c][i];
			}
		}
		
		inputOffset += inputSamples;
		outputOffset += blockSize;
	}
	endTime=__rdtsc();
	printf("Tempo impiegato per l'elaborazione (in cicli di clock): %ld\n",(endTime-startTime));
	if (!outputWav.write(outputFile)) 
	{
		printf(outputWav.result.reason.c_str());
		exit(-1);
	}
}

int main(int argc, char **argv) {
	if(argc!=4)
	{
		printf("Sintassi: %s <inputFile> <outputFile> <semitoni>\n",argv[0]);
		exit(-1);
	}
	pitchShifter(argv[1],argv[2],atof(argv[3]));
}
