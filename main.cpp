#include "shift-stretch.h"
#include "util/wav.h"
#include <stdio.h>
#include <math.h>
class WavCmd {
	std::string inputFile, outputFile;
	Wav inputWav, outputWav;
	
	double timeFactor = 1, freqFactor = 1;
	double blockMs = 80, overlapFactor = 4, searchMs = 10;

	template<class Processor>
	void processBlocks(Processor &processor, double stretchFactor) {
		int channels = inputWav.channels;
		int blockSize = 256;
		
		std::vector<std::vector<double>> inputBuffers(channels), outputBuffers(channels);
		std::vector<double *> inputPointers(channels), outputPointers(channels);
		for (auto &b : outputBuffers) b.resize(blockSize);
		
		outputWav.channels = inputWav.channels;
		int inputOffset = 0, outputOffset = 0;
		int inputLength = int(inputWav.length());
		int totalLatency = std::round(processor.inputLatency()*stretchFactor + processor.outputLatency());
		int outputLength = inputWav.length()*stretchFactor;
		while (outputOffset < outputLength + totalLatency*2) {
			// For `blockSize` output samples, how many input samples should we have?
			int inputSamples = processor.samplesForOutput(blockSize);
			// Make sure our input buffers are large enough
			if (inputSamples > int(inputBuffers[0].size())) {
				for (auto &b : inputBuffers) b.resize(inputSamples);
			}
			// Fill them up
			for (int c = 0; c < channels; ++c) {
				for (int i = 0; i < inputSamples; ++i) {
					// Fill input either from WAV, or with 0
					if (inputOffset + i < inputLength) {
						inputBuffers[c][i] = inputWav[c][inputOffset + i];
					} else {
						inputBuffers[c][i] = 0;
					}
				}
				inputPointers[c] = inputBuffers[c].data();
				outputPointers[c] = outputBuffers[c].data();
			}
			
			processor.process(inputPointers.data(), inputSamples, outputPointers.data(), blockSize);
			
			outputWav.samples.resize((outputOffset + blockSize)*channels);
			for (int c = 0; c < channels; ++c) {
				for (int i = 0; i < blockSize; ++i) {
					outputWav[c][outputOffset + i] = outputBuffers[c][i];
				}
			}
			
			inputOffset += inputSamples;
			outputOffset += blockSize;
		}
	}
public:
	WavCmd(char* inputFile, char* outputFile, double semitones) {
		freqFactor = pow(2,semitones/12);
		bool fixedPhase = false;

		if (!inputWav.read(inputFile)) 
		{
			printf(inputWav.result.reason.c_str());
			exit(-1);
		}
		outputWav.channels = inputWav.channels;
		outputWav.sampleRate = inputWav.sampleRate;

		int blockSamples = int(blockMs*0.001*inputWav.sampleRate + 0.5);
		int intervalSamples = int(blockSamples/overlapFactor);
		int searchSamples = int(searchMs*0.001*inputWav.sampleRate + 0.5);
		SpectralCutStretch stretch(fixedPhase);
		stretch.configure(inputWav.channels, blockSamples, intervalSamples);
		stretch.setTimeFactor(timeFactor);
		stretch.setFreqFactor(freqFactor);
		processBlocks(stretch, timeFactor);
		if (!outputWav.write(outputFile)) 
		{
			printf(outputWav.result.reason.c_str());
			exit(-1);
		}
	}
};

int main(int argc, char **argv) {
	if(argc!=4)
	{
		printf("Sintassi: %s <inputFile> <outputFile> <semitoni>\n",argv[0]);
	}
	WavCmd cmd(argv[1],argv[2],atof(argv[3]));
}
