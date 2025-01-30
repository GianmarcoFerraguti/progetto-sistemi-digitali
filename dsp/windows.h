#ifndef SIGNALSMITH_DSP_WINDOWS_H
#define SIGNALSMITH_DSP_WINDOWS_H
#include <math.h>
float beta;
float invB0;
inline static float bessel0(float x) {
	const float significanceLimit = 1e-4;
	float result = 0;
	float term = 1;
	float m = 0;
	while (term > significanceLimit) {
		result += term;
		++m;
		term *= (x*x)/(4*m*m);
	}
	return result;
}
float bandwidthToBeta(float bandwidth) {
	//Idea carina ma non essenziale: raggruppare alcuni termini della seguente espressione in un registro esteso in modo da calcolarli contemporanemante
	bandwidth = bandwidth + 8/((bandwidth + 3)*(bandwidth + 3)) + 0.25*std::max(3 - bandwidth, 0.0F); //heuristicBandwidth
	float alpha = std::sqrt(bandwidth*bandwidth*0.25f - 1);
	return alpha*M_PI;
}
void kaiserWithBandwidth(float bandwidth) {
	beta = bandwidthToBeta(bandwidth);
	invB0 = 1/bessel0(beta);
}
void kaiserFill(float *&data, int size) {
	float invSize = 1.0/size;
	//Si potrebbe riscrivere in SIMD
	for (int i = 0; i < size; ++i) {
		float r = (2*i + 1)*invSize - 1;
		float arg = std::sqrt(1 - r*r);
		data[i] = bessel0(beta*arg)*invB0;
	}
}
#endif
