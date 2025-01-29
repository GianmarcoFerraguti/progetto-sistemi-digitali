#ifndef SIGNALSMITH_DSP_WINDOWS_H
#define SIGNALSMITH_DSP_WINDOWS_H
#include <math.h>
	class Kaiser {
	public:
		__bfloat16 beta;
		__bfloat16 invB0;

		inline static __bfloat16 bessel0(__bfloat16 x) {
			const __bfloat16 significanceLimit = 1e-4;
			__bfloat16 result = 0;
			__bfloat16 term = 1;
			__bfloat16 m = 0;
			while (term > significanceLimit) {
				result += term;
				++m;
				term *= (x*x)/(4*m*m);
			}

			return result;
		}
		Kaiser(__bfloat16 beta) : beta(beta), invB0(1/bessel0(beta)) {}
		static Kaiser withBandwidth(__bfloat16 bandwidth) {
			return Kaiser(bandwidthToBeta(bandwidth));
		}

		static __bfloat16 bandwidthToBeta(__bfloat16 bandwidth) {
			//Idea carina ma non essenziale: raggruppare alcuni termini della seguente espressione in un registro esteso in modo da calcolarli contemporanemante
			bandwidth = bandwidth + 8/((bandwidth + 3)*(bandwidth + 3)) + 0.25*std::max((float)(3 - bandwidth), 0.0F); //heuristicBandwidth
			__bfloat16 alpha = std::sqrt((float)(bandwidth*bandwidth*0.25) - 1);
			return alpha*M_PI;
		}

		void fill(__bfloat16 *&data, int size) const {
			__bfloat16 invSize = 1.0/size;
			//Si potrebbe riscrivere in SIMD
			for (int i = 0; i < size; ++i) {
				__bfloat16 r = (2*i + 1)*invSize - 1;
				__bfloat16 arg = std::sqrt((float)(1 - r*r));
				data[i] = bessel0(beta*arg)*invB0;
			}
		}
	};
#endif
