#ifndef SIGNALSMITH_DSP_WINDOWS_H
#define SIGNALSMITH_DSP_WINDOWS_H

	class Kaiser {
	public:
		double beta;
		double invB0;

		inline static double bessel0(double x) {
			const double significanceLimit = 1e-4;
			double result = 0;
			double term = 1;
			double m = 0;
			while (term > significanceLimit) {
				result += term;
				++m;
				term *= (x*x)/(4*m*m);
			}

			return result;
		}
		Kaiser(double beta) : beta(beta), invB0(1/bessel0(beta)) {}
		static Kaiser withBandwidth(double bandwidth) {
			return Kaiser(bandwidthToBeta(bandwidth));
		}

		static double bandwidthToBeta(double bandwidth) {
			//Idea carina ma non essenziale: raggruppare alcuni termini della seguente espressione in un registro esteso in modo da calcolarli contemporanemante
			bandwidth = bandwidth + 8/((bandwidth + 3)*(bandwidth + 3)) + 0.25*std::max(3 - bandwidth, 0.0); //heuristicBandwidth
			double alpha = std::sqrt(bandwidth*bandwidth*0.25 - 1);
			return alpha*M_PI;
		}

		void fill(double *&data, int size) const {
			double invSize = 1.0/size;
			//Si potrebbe riscrivere in SIMD
			for (int i = 0; i < size; ++i) {
				double r = (2*i + 1)*invSize - 1;
				double arg = std::sqrt(1 - r*r);
				data[i] = bessel0(beta*arg)*invB0;
			}
		}
	};

	void forcePerfectReconstruction(double *&data, int windowLength, int interval) {
		//Si potrebbe riscrivere in SIMD
		for (int i = 0; i < interval; ++i) {
			double sum2 = 0;
			for (int index = i; index < windowLength; index += interval) {
				sum2 += data[index]*data[index];
			}
			double factor = 1/std::sqrt(sum2);
			for (int index = i; index < windowLength; index += interval) {
				data[index] *= factor;
			}
		}
	}
#endif
