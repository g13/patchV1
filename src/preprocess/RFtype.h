#ifndef RFTYPE_H
#define RFTYPE_H
#include <vector>
#include <numeric>
#include "../types.h"
#include "../util/util.h"
#include <cassert>
#include <type_traits>
//#include <algorithm>

#define nRFtype 5
#define nOutputType 4
#define nInputType 6

enum class RFtype: PosInt {
	nonOppopent_gabor = 0,  // different cone input correlates, i.e., have the same sign, in gabor form, find in V1
	nonOppopent_cs = 1,  // different cone input correlates, i.e., have the same sign, in center surround form not necessarily concentric, find in V1
	singleOppopent = 2, // center surround cone opponency, LGN and V1
	doubleOppopent_cs = 3, // center surround cone and spatial opponency, V1
    doubleOppopent_gabor = 4 // cone and spatial opponency in Gabor profile, V1
};

enum class OutputType: PosInt { // V1 local RF subregion
	// if RF has double peaks, choose the first on the left
	// surround ignored when RF have double peaks
	// non-opponent
	LonMon = 0,
	LoffMoff = 1,
	// opponent
	LonMoff = 2,
	LoffMon = 3
};

struct InputActivation {
    Float actPercent[nInputType];
    __host__ __device__
    __forceinline__ InputActivation() {};

    __host__ __device__
    __forceinline__ InputActivation(Float percent[]) {
        for (PosInt i=0; i<nInputType; i++) {
            actPercent[i] = percent[i];
        }
    }

    __host__ __device__
    __forceinline__ void assign(Float percent[]) {
        for (PosInt i=0; i<nInputType; i++) {
            actPercent[i] = percent[i];
        }
    }
};

enum class InputType: PosInt { // LGN
    // center-surround
    LonMoff = 0, // parvocellular
    LoffMon = 1,
    MonLoff = 2,
    MoffLon = 3,
    OnOff = 4, // magnocellular
    OffOn = 5
    // ignore mixed surround, almost always from a different cone type
};

typedef std::underlying_type<RFtype>::type RFtype_t;
typedef std::underlying_type<InputType>::type InputType_t;
typedef std::underlying_type<OutputType>::type OutputType_t;

// normalize x to [-1,1], y to baRatio*[-1,1]
inline std::pair<Float, Float> transform_coord_to_unitRF(Float x, Float y, const Float mx, const Float my, const Float theta, const Float a) {
    // a is half-width at the x-axis
    x = (x - mx)/a;
    y = (y - my)/a;
    Float new_x, new_y;
    new_x = cos(theta) * x + sin(theta) * y;
	new_y = -sin(theta) * x + cos(theta) * y;
    return std::make_pair(new_x, new_y);
}

// TODO: document type meanings
inline Int match_OnOff(InputType iType, OutputType oType, Float &modulation) {
	Int match;
   	switch (oType) {
   	    case OutputType::LonMon:
   			switch (iType) {
                case InputType::LonMoff: case InputType::MonLoff: case InputType::OnOff: match = 1;
   					break;
                case InputType::LoffMon: case InputType::MoffLon: case InputType::OffOn: match = -1;
					break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
   	        break;
   	    case OutputType::LoffMoff:
   	        switch (iType) {
   				case InputType::LoffMon: case InputType::MoffLon: case InputType::OffOn: match = 1;
   					break;                                                               
   				case InputType::LonMoff: case InputType::MonLoff: case InputType::OnOff: match = -1;
					break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
			break;
   	    default: throw("There's no implementation of such non-opponent RF");
	}
   	return match;
}

inline Int oppose_Cone_OnOff_double(InputType iType, OutputType oType, Float &modulation) {
	Int opponent;
   	switch (oType) {
   	    case OutputType::LonMoff:
   			switch (iType) {
   				case InputType::LonMoff: case InputType::MoffLon: opponent = 1;
   					break;
   				case InputType::LoffMon: case InputType::MonLoff: opponent = -1;
					break;
                case InputType::OnOff: case InputType::OffOn: opponent = 0;
                    break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
   	        break;
   	    case OutputType::LoffMon:
   	        switch (iType) {
   				case InputType::LoffMon: case InputType::MonLoff: opponent = 1;
   					break;
   				case InputType::LonMoff: case InputType::MoffLon: opponent = -1;
					break;
                case InputType::OnOff: case InputType::OffOn: opponent = 0;
                    break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
			break;
   	    default: throw("There's no implementation of such opponent RF");
	}
   	return opponent;
}

inline Int oppose_Cone_OnOff_single(InputType iType, OutputType oType, Float &modulation) {
	Int opponent;
   	switch (oType) {
   	    case OutputType::LonMoff:
   			switch (iType) {
   				case InputType::LonMoff: opponent = 1;
   					break;
                case InputType::LoffMon: case InputType::MoffLon: case InputType::MonLoff: case InputType::OffOn: opponent = -1;
					break;
                case InputType::OnOff: opponent = 0;
                    break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
   	        break;
   	    case OutputType::LoffMon:
   	        switch (iType) {
   				case InputType::LoffMon: opponent = 1;
   					break;
                case InputType::LonMoff: case InputType::MonLoff: case InputType::MoffLon: case InputType::OnOff: opponent = -1;
					break;
                case InputType::OffOn: opponent = 0;
                    break;
				default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        }
			break;
   	    default: throw("There's no implementation of such opponent RF");
	}
   	return opponent;
}

struct LinearReceptiveField { // RF sample without implementation of check_opponency
    RFtype rfType; // class identifier for pointer
	std::vector<Float> prob;
    Size n;
    //                        a is the minor-axis
    Float sfreq, phase, amp, theta, a, baRatio, sig;
    OutputType oType;
	bool strictStrength;
	/* not needed
    // default constructor
    LinearReceptiveField(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, Float sig = 1.177):
        n(n),
        sfreq(sfreq),
        phase(phase),
        amp(amp),
        theta(theta),
        a(a),
        baRatio(baRatio),
        oType(oType),
        sig(sig)
    {} */
    virtual void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, bool strictStrength, Float sig = 1.177) {
		assert(amp <= 1);
        this->n = n;
        this->sfreq = sfreq;
        this->phase = phase;
        this->amp = amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
		this->strictStrength = strictStrength;
	}

	virtual void clear() {
		prob.clear();
	}
 
    virtual Size construct_connection_N(std::vector<Float> &x, std::vector<Float> &y, std::vector<InputType> &iType, std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen, Float fnLGNeff, bool p_n, Size max_nCon, bool top_pick, Float thres = 0.5) {
        Size nConnected;
        if (n > 0) {
            if (fnLGNeff > 0) {
		        prob.reserve(n);
                // putative RF center
			    Float cx, cy;
                std::tie(cx, cy) = average2D<Float>(x, y);
		        for (Size i = 0; i < n; i++) {
                    Float norm_x, norm_y;
                    // orient and normalize LGN coord
		            std::tie(norm_x, norm_y) = transform_coord_to_unitRF(x[i], y[i], cx, cy, theta, a);
                    // calc. prob. dist. distance-dependent envelope at coord.
                    Float envelope = get_envelope(norm_x, norm_y, top_pick, thres, sig);
                    // calc. modulation of prob at coord.
                    Float modulation = modulate(norm_x, norm_y);
                    /* TEST with no modulation: bool RefShift = false;
                    if (modulation < 0.5) {
                        RefShift = true;
                    } */
                    // calc. cone and position opponency at coord.
	                Float opponent = check_opponency(iType[i], modulation);
					Float _prob = get_prob(opponent, modulation, envelope);
					assert(_prob >= 0); 
                    prob.push_back(_prob);
                    /* TEST with no modulation: if (opponent < 0.0 && RefShift || (opponent < 0.0 && (rfType == RFtype::doubleOppopent_cs || rfType == RFtype::singleOppopent || rfType == RFtype::nonOppopent_cs))) {
                        if (prob.back() > 0.0) {
                            std::cout << envelope << " * (0.5 + " << amp << " * " << opponent << " * " << modulation <<  ") = " << prob.back() << "\n";
                            assert(prob.back() == 0.0);
                        }
                    } */
                }
				// set the number of connections
                Size sSum = normalize(fnLGNeff, p_n);
                // make connection and update ID and strength list
                nConnected = connect_N(idList, strengthList, rGen, max_nCon, sSum, top_pick);
            } else {
                idList = std::vector<Size>();
                idList.shrink_to_fit();
                nConnected = 0;
            }
        }  else {
            nConnected = 0;
        }
        return nConnected;
    }
    // Probability envelope based on distance
    //virtual Float get_envelope(Float x, Float y, bool top_pick, Float thres = 0.5, Float sig = 1.1775) {
    virtual Float get_envelope(Float x, Float y, bool top_pick, Float thres = 0.5, Float sig = 0.6) {
        Float v = exp(-0.5*(pow(x/sig,2)+pow(y/(sig*baRatio),2)));
		//if (top_pick) {
		//	if (v > thres) {
		//		v = 1.0;
		//	}
		//}
        return v;
        //return 1.0;
        // baRatio comes from Dow et al., 1981
    }
    // Full cosine modulation, modulation is on cone and on-off types
    virtual Float modulate(Float x, Float y) {
        return 0.5 + 0.5 * cos(sfreq*(x + phase)*M_PI);
		//assert(abs(x) <= 1.0);
		//assert(sfreq*x * M_PI + phase > 0 && sfreq*x * M_PI + phase < 2*M_PI);
        // sfreq should be given as a percentage of the width
        // when sfreq == 1, the RF contain a full cycle of cos(x), x\in[-pi, pi]
    }
    // To be implemented by derived class
    virtual Float check_opponency(InputType iType, Float &modulation) = 0;
    // produce connection prob. at coord. for i-th LGN
    virtual Float get_prob(Float opponent, Float modulation, Float envelope) {
        return envelope * (1.0 + amp * opponent * modulation);
    }
    // normalize prob.
    Size normalize(Float fnLGNeff, bool p_n) {
	    // average connection probability is controlled at fnLGNeff.
		Float norm;
		Size sSum;
        if (p_n) { // if percentage
			sSum = static_cast<Size>(round(fnLGNeff*prob.size()));
        } else { // number restriction
			sSum = static_cast<Size>(round(fnLGNeff));
        }
		if (sSum > 0) {
    		norm = std::accumulate(prob.begin(), prob.end(), 0.0) / sSum;
	    	//std::cout << "norm = " << norm << "\n";
	    	//print_list<Float>(prob);
        	//Float sum = 0.0;
        	if (norm == 0) {
				sSum = 0;
			} else {
				Size nAvail = 0;
        		for (PosInt i=0; i<prob.size(); i++) {
					if (prob[i] > 0) {
						nAvail++;
					}
        		    prob[i] = prob[i] / norm;
        		}
				sSum = nAvail < sSum? nAvail: sSum;
			}
		}
    	//std::cout << fnLGNeff*prob.size() << " ~ " << sum << "\n";
		return sSum;
    }
    // make connections
    virtual Size connect_N(std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen, Size max_nCon, Size sSum, bool top_pick) {
		// make connections and normalized strength i.e., if prob > 1 then s = 1 else s = prob
        std::uniform_real_distribution<Float> uniform(0,1);
        std::normal_distribution<Float> normal(0,0.05);
		strengthList.reserve(n);
		std::vector<Size> newList;
		newList.reserve(n);
		if (sSum > 0) {
			Size count = 0; 
			Size n_to_connect = sSum;
			assert(n_to_connect <= max_nCon);
			if (!top_pick) {
				do  { // variable connection
					newList.clear();
					strengthList.clear();
					for (PosInt i = 0; i < n; i++) {
						if (uniform(rGen) < prob[i]) {
							newList.push_back(idList[i]);
							if (prob[i] > 1) {
								strengthList.push_back(prob[i]);
							} else {
								strengthList.push_back(1);
							}
						}
					}
					count++;
					if (count > 20) {
						std::cout << "too many iters, sum over prob #" << prob.size() << " = " << std::accumulate(prob.begin(), prob.end(), 0.0) << "\n";
						assert(count <= 20);
					}
				} while (newList.size() == 0 || newList.size() > max_nCon);
			} else { // fixed number
				/*
				do  { // decide number to connect first
					n_to_connect = 0;
					for (PosInt i = 0; i < n; i++) {
						prob[i] *= 1+normal(rGen);
						if (uniform(rGen) < prob[i]) {
							n_to_connect++;
						}
					}
					count++;
					if (count > 20) {
						std::cout << "too many iters, sum over prob #" << prob.size() << " = " << std::accumulate(prob.begin(), prob.end(), 0.0) << "\n";
						assert(count <= 20);
					}
				} while (n_to_connect > max_nCon || n_to_connect == 0);
				*/
				// pick the tops
				do  {
					// get the LGN with max prob
					PosInt i = std::distance(prob.begin(), std::max_element(prob.begin(), prob.end()));
					newList.push_back(idList[i]);
					// put the max prob to min
					prob[i] = -prob[i];
					if (prob[i] > 1) {
						strengthList.push_back(prob[i]);
					} else {
						strengthList.push_back(1);
					}
				} while (newList.size() < n_to_connect || newList.size() > max_nCon);
			}

        	Float con_irl = std::accumulate(strengthList.begin(), strengthList.end(), 0.0);
			if (strictStrength) {
        		Float ratio = sSum/con_irl;
        		for (PosInt i=0; i<strengthList.size(); i++) {
        		    strengthList[i] *= ratio; 
        		}
			}
		}
		idList = newList;
		idList.shrink_to_fit();
        return static_cast<Size>(idList.size());
    }

    virtual Size connect_thres(std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen, Float conThres, Size max_nCon, std::vector<Int> &iPick, std::vector<Float> &norm_x, std::vector<Float> &norm_y, Size n, PosInt iV1) {
		// make connections and normalized strength i.e., if prob > 1 then s = 1 else s = prob

		std::vector<Size> list0;
		list0.reserve(n);
		std::vector<Float> strList0;
		strList0.reserve(n);

		std::vector<Float> x;
		std::vector<Float> y;
		std::vector<Int> pick2;
		Int count = 0;
		Float probSum = 0.0;
		for (PosInt i = 0; i < n; i++) {
			if (prob[i] > conThres) {
				list0.push_back(idList[i]);
				//if (prob[i] > 1) {
					strList0.push_back(prob[i]);
				//} else {
				//	strList0.push_back(1);
				//}
				count++;
				pick2.push_back(iPick[i]);
				x.push_back(norm_x[i]);
				y.push_back(norm_y[i]);
				probSum += prob[i];
			}
		}
		Size m = list0.size();

		// pickout extra LGN cons with equal prob.
		std::vector<bool> ipick(m, true);
        std::uniform_real_distribution<Float> uniform(0,1);
		if (count > max_nCon) {
			while (count-max_nCon > 0) {
				PosInt ipicked = static_cast<int>(flooring(uniform(rGen) * m));
				if (ipick[ipicked]) {
					ipick[ipicked] = false;
					count--;
				}
			}
			strengthList.reserve(max_nCon);
			idList.clear();
			idList.reserve(max_nCon);
		} else {
			strengthList.reserve(count);
			idList.clear();
			idList.reserve(count);
		}
		// push to output vectors
		std::vector<Float> cx(2, 0), cy(2, 0);
		Size n0 = 0;
		Size n1 = 0;
		for (PosInt i=0; i<m; i++) {
			if (ipick[i]) {
				idList.push_back(list0[i]);
				strengthList.push_back(strList0[i]);
				if (pick2[i] > 0) {
					cx[1] += x[i];
					cy[1] += y[i];
					n1++;

				} else {
					cx[0] += x[i];
					cy[0] += y[i];
					n0++;
				}
			}
		}
		norm_x.clear();
		norm_y.clear();
		norm_x = x;
		norm_y = y;
		norm_x.shrink_to_fit();
		norm_y.shrink_to_fit();
		Size nConnected = idList.size();
		assert(n0 + n1 == nConnected);
		if (n0 > 0 && n1 > 0) {
			cx[0] /= n0;
			cy[0] /= n0;
			cx[1] /= n1;
			cy[1] /= n1;
			Float dis = square_root((cx[0] - cx[1]) * (cx[0] - cx[1]) + (cy[0] - cy[1]) * (cy[0] - cy[1]));
			if (dis >= 2) {
				std::cout << "dis = " << dis  << "\n";
				std::cout << " c0 = [" << cx[0] << ", " << cy[0] << "]\n";
				std::cout << " c1 = [" << cx[1] << ", " << cy[1] << "]\n";
				//assert(dis < 2);
			}
			sfreq = 1/(dis*2/2*2*a);
		} else {
			sfreq = 0;
		}
		//std::cout << "sfreq = " << sfreq << " from " << nConnected << " LGN cells\n";

		assert(nConnected <= max_nCon);

        return nConnected;
    }

    virtual Size construct_connection_thres(std::vector<Float> &x, std::vector<Float> &y, std::vector<InputType> &iType, std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen, Size max_nCon, Float conThres, Float zero, Float &true_sfreq, Float vx, Float vy, PosInt iV1) {
        Size nConnected;
        if (n > 0) {
            if (zero > 0) {
        		std::uniform_real_distribution<Float> uniform(0,1);
		        prob.reserve(n);
                // putative RF center
			    Float cx, cy;
                std::tie(cx, cy) = average2D<Float>(x, y);
				//cx = vx;
				//cy = vy;
				std::vector<Float> norm_x;
				std::vector<Float> norm_y; 
				std::vector<Int> iPick;  // 0 Lon,Moff,On, 1 Loff,Mon,Off
				std::vector<Float> envelope_value; 
				std::vector<Size> newList; 
				std::vector<InputType> newType; 

				Int iOnOff;
				switch (oType) {
					case OutputType::LonMoff: case OutputType::LonMon: iOnOff = 1;
   						break;
					case OutputType::LoffMon: case OutputType::LoffMoff: iOnOff = -1;
   						break;
				}
		        for (Size i=0; i<n; i++) {
					Float temp_x, temp_y;
		        	std::tie(temp_x, temp_y) = transform_coord_to_unitRF(x[i], y[i], cx, cy, theta, a);

					Float temp_value = get_envelope(temp_x, temp_y, false);
                    if (temp_value > 0.5) {
						norm_x.push_back(temp_x);
						norm_y.push_back(temp_y);
						envelope_value.push_back(temp_value);
						//envelope_value.push_back(1.0);
   						switch (iType[i]) {
							case InputType::LonMoff: case InputType::MoffLon: case InputType::OnOff: iPick.push_back(1);
   								break;
   							case InputType::LoffMon: case InputType::MonLoff: case InputType::OffOn: iPick.push_back(-1);
								break;
							default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        			}
						newList.push_back(idList[i]);
						newType.push_back(iType[i]);
					}
				}
				// get a LGN cell's position as the phase, randomly chose based on the con prob from their amplitude modulation over VF distance
				Float rand = uniform(rGen);
				PosInt m = iPick.size();
				PosInt iPhase = static_cast<PosInt>(flooring(rand*m));

				phase = norm_x[iPhase]*M_PI + iPick[iPhase]*iOnOff*M_PI/2;
				//phase = M_PI/2.0;
                
				Float max_prob[2] = {0, 0};
		        for (Size i=0; i<m; i++) {
                    // calc. modulation of prob at coord.
                    Float modulation = modulate(norm_x[i], norm_y[i]); // sfreq, phase
                    // calc. cone and position opponency at coord.
	                Float opponent = check_opponency(newType[i], modulation);
					Float _prob = get_prob(opponent, modulation, envelope_value[i]);
					assert(_prob >= 0);
                    prob.push_back(_prob);
					if (iPick[i] == 1) {
						if (_prob > max_prob[0]) max_prob[0] = _prob;
					} else {
						if (_prob > max_prob[1]) max_prob[1] = _prob;
					}
                }
		        for (Size i=0; i<m; i++) {
					if (iPick[i] == 1) {
						prob[i] /= max_prob[0];
					} else {
						prob[i] /= max_prob[1];
					}
				}
                // make connection and update ID and strength list
                nConnected = connect_thres(newList, strengthList, rGen, conThres, max_nCon, iPick, norm_x, norm_y, m, iV1);
				idList = newList; 
				idList.shrink_to_fit();
				true_sfreq = sfreq;

				Float norm_cx, norm_cy;
                std::tie(norm_cx, norm_cy) = average2D<Float>(norm_x, norm_y);
				Float center_deviation = square_root(norm_cx* norm_cx + norm_cy*norm_cy);
				//if (center_deviation > 0.25) {
				//	std::cout << "c_d = " << center_deviation << "\n";
				//	std::cout << "from " << n << " to " << m << " to " << norm_x.size() << "\n";
				//}
            } else {
                idList = std::vector<Size>();
                idList.shrink_to_fit();
                nConnected = 0;
				true_sfreq = 0;
            }
        }  else {
            nConnected = 0;
			true_sfreq = 0;
        }
        return nConnected;
    }

    virtual Size construct_connection_opt(std::vector<Float> &x, std::vector<Float> &y, std::vector<InputType> &iType, std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen, Size max_nCon, Float conThres, Float zero, Float &true_sfreq, Float vx, Float vy, PosInt iV1) {
        Size nConnected;
        if (n > 0) {
            if (zero > 0) {
        		std::uniform_real_distribution<Float> uniform(0,1);
		        prob.reserve(n);
                // putative RF center
			    Float cx, cy;
                std::tie(cx, cy) = average2D<Float>(x, y);
				//cx = vx;
				//cy = vy;
				std::vector<Float> norm_x;
				std::vector<Float> norm_y; 
				std::vector<Int> iPick;  // 0 Lon,Moff,On, 1 Loff,Mon,Off
				std::vector<Float> envelope_value; 
				std::vector<Size> newList; 
				std::vector<InputType> newType;

				Int iOnOff;
				switch (oType) {
					case OutputType::LonMoff: case OutputType::LonMon: iOnOff = 1;
   						break;
					case OutputType::LoffMon: case OutputType::LoffMoff: iOnOff = -1;
   						break;
				}
		        for (Size i=0; i<n; i++) {
					Float temp_x, temp_y;
		        	std::tie(temp_x, temp_y) = transform_coord_to_unitRF(x[i], y[i], cx, cy, theta, a);

					Float temp_value = get_envelope(temp_x, temp_y, false);
                    if (temp_value > 0.5) {
						norm_x.push_back(temp_x);
						norm_y.push_back(temp_y);
						envelope_value.push_back(temp_value);
   						switch (iType[i]) {
							case InputType::LonMoff: case InputType::MoffLon: case InputType::OnOff: iPick.push_back(1);
   								break;
   							case InputType::LoffMon: case InputType::MonLoff: case InputType::OffOn: iPick.push_back(-1);
								break;
							default: throw("There's no implementation of such combination of cone types for center-surround RF");
   	        			}
						newList.push_back(idList[i]);
						newType.push_back(iType[i]);
					}
				}
				// get a LGN cell's position as the phase, randomly chose based on the con prob from their amplitude modulation over VF distance
				PosInt m = iPick.size();
                Float phase = iOnOff/2; // iOnOff = 1, positive peak to the left of origin
		        for (Size i=0; i<m; i++) {
                    // calc. modulation of prob at coord.
                    Float modulation = modulate(norm_x[i], norm_y[i]); // sfreq, phase
                    // calc. cone and position opponency at coord.
	                Float opponent = check_opponency(newType[i], modulation);
					Float _prob = get_prob(opponent, modulation, envelope_value[i]);
					assert(_prob >= 0);
                    prob.push_back(_prob);
                }
                // make connection and update ID and strength list
                nConnected = connect_opt(newList, strengthList, rGen, max_nCon, iPick, norm_x, norm_y, m, iV1);
				idList = newList; 
				idList.shrink_to_fit();
				true_sfreq = sfreq;

				Float norm_cx, norm_cy;
                std::tie(norm_cx, norm_cy) = average2D<Float>(norm_x, norm_y);
				Float center_deviation = square_root(norm_cx* norm_cx + norm_cy*norm_cy);
				//if (center_deviation > 0.25) {
				//	std::cout << "c_d = " << center_deviation << "\n";
				//	std::cout << "from " << n << " to " << m << " to " << norm_x.size() << "\n";
				//}
            } else {
                idList = std::vector<Size>();
                idList.shrink_to_fit();
                nConnected = 0;
				true_sfreq = 0;
            }
        }  else {
            nConnected = 0;
			true_sfreq = 0;
        }
        return nConnected;
    }

    virtual Size connect_opt(std::vector<Size> &idList, std::vector<Float> &strengthList, RandomEngine &rGen, Size max_nCon, std::vector<Int> &iPick, std::vector<Float> &norm_x, std::vector<Float> &norm_y, Size n, PosInt iV1) {
		// make connections and normalized strength i.e., if prob > 1 then s = 1 else s = prob

		std::vector<Size> list0;
		list0.reserve(n);
		std::vector<Float> strList0;
		strList0.reserve(n);

		std::vector<Float> xon;
		std::vector<Float> yon;
		std::vector<Float> xoff;
		std::vector<Float> yoff;
		std::vector<Int> pick2;
		Int count = 0;
		Float probSum = 0.0;
        
        // get the first one:
		PosInt i = std::distance(prob.begin(), std::max_element(prob.begin(), prob.end()));
        list0.push_back(idList[i]);
		strList0.push_back(prob[i]);
        prob[i] = 0;

        if (iPick[i] > 0) {
			xon.push_back(norm_x[i]);
			yon.push_back(norm_y[i]);
        } else {
			xoff.push_back(norm_x[i]);
			yoff.push_back(norm_y[i]);
        }
		pick2.push_back(iPick[i]);

		for (PosInt i = 0; i < n; i++) {
			list0.push_back(idList[i]);
			strList0.push_back(prob[i]);
			count++;
			pick2.push_back(iPick[i]);
			probSum += prob[i];
		}
		Size m = list0.size();

		// pickout extra LGN cons with equal prob.
		std::vector<bool> ipick(m, true);
        std::uniform_real_distribution<Float> uniform(0,1);
		if (count > max_nCon) {
			while (count-max_nCon > 0) {
				PosInt ipicked = static_cast<int>(flooring(uniform(rGen) * m));
				if (ipick[ipicked]) {
					ipick[ipicked] = false;
					count--;
				}
			}
			strengthList.reserve(max_nCon);
			idList.clear();
			idList.reserve(max_nCon);
		} else {
			strengthList.reserve(count);
			idList.clear();
			idList.reserve(count);
		}
		// push to output vectors
		std::vector<Float> cx(2, 0), cy(2, 0);
		Size n0 = 0;
		Size n1 = 0;
		for (PosInt i=0; i<m; i++) {
			if (ipick[i]) {
				idList.push_back(list0[i]);
				strengthList.push_back(strList0[i]);
				if (pick2[i] > 0) {
					cx[1] += x[i];
					cy[1] += y[i];
					n1++;
				} else {
					cx[0] += x[i];
					cy[0] += y[i];
					n0++;
				}
			}
		}
		norm_x.clear();
		norm_y.clear();
		norm_x = x;
		norm_y = y;
		norm_x.shrink_to_fit();
		norm_y.shrink_to_fit();
		Size nConnected = idList.size();
		assert(n0 + n1 == nConnected);
		if (n0 > 0 && n1 > 0) {
			cx[0] /= n0;
			cy[0] /= n0;
			cx[1] /= n1;
			cy[1] /= n1;
			Float dis = square_root((cx[0] - cx[1]) * (cx[0] - cx[1]) + (cy[0] - cy[1]) * (cy[0] - cy[1]));
			if (dis >= 2) {
				std::cout << "dis = " << dis  << "\n";
				std::cout << " c0 = [" << cx[0] << ", " << cy[0] << "]\n";
				std::cout << " c1 = [" << cx[1] << ", " << cy[1] << "]\n";
				//assert(dis < 2);
			}
			sfreq = 1/(dis*2/2*2*a);
		} else {
			sfreq = 0;
		}
		//std::cout << "sfreq = " << sfreq << " from " << nConnected << " LGN cells\n";

		assert(nConnected <= max_nCon);

        return nConnected;
    }
};

struct SingleOpponent: LinearReceptiveField {
	// center-surround
    SingleOpponent() {
        rfType = RFtype::singleOppopent;
    }
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, bool strictStrength, Float sig = 1.177) override {
        this->n = n;
        this->sfreq = 0.0;
        this->phase = 0.0;
        this->amp = 0.5*amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}

    Float modulate(Float x, Float y) override {
        return 1.0;
    }

    Float check_opponency(InputType iType, Float &modulation) override {
		return 1.0 * oppose_Cone_OnOff_single(iType, oType, modulation);
	}

    Float get_prob(Float opponent, Float modulation, Float envelope) override {
        return envelope * (0.5 + amp * opponent * modulation);
    }
};

struct DoubleOpponent_Gabor: LinearReceptiveField {
	// Gabor
    DoubleOpponent_Gabor() {
        rfType = RFtype::doubleOppopent_gabor;
    }
    Float check_opponency(InputType iType, Float &modulation) override {
        Float opponent;
        OutputType currType;
		if (modulation < 0.5) {
			// switch to the dominant type
			modulation = 1 - modulation;
			currType = static_cast<OutputType> (5 - static_cast<Size>(oType));
			assert(currType == OutputType::LonMoff || currType == OutputType::LoffMon);
        } else {
            currType = oType;
        }
		opponent = 1.0 * oppose_Cone_OnOff_double(iType, currType, modulation);
        return opponent;
    }

    Float get_prob(Float opponent, Float modulation, Float envelope) {
        Float v = (1.0 + amp * opponent * modulation)/2;
        if (v < 0.5) {
            v = 0;
        }
        v = envelope * v;

	    assert(!isnan(v));
        return v;
    }
};

struct DoubleOpponent_CS: LinearReceptiveField {
	// center-surround
    DoubleOpponent_CS() {
        rfType = RFtype::doubleOppopent_cs;
    }
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, bool strictStrength, Float sig = 1.177) override {
        this->n = n;
        this->sfreq = 0.0;
        this->phase = 0.0;
        this->amp = 0.5*amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}

    Float modulate(Float x, Float y) override {
        return 1.0;
    }

    Float check_opponency(InputType iType, Float &modulation) override {
		return 1.0 * oppose_Cone_OnOff_double(iType, oType, modulation);
	}

    Float get_prob(Float opponent, Float modulation, Float envelope) override {
        return envelope * (0.5 + amp * opponent * modulation);
    }
};

struct NonOpponent_Gabor: LinearReceptiveField {
	// Gabor
    NonOpponent_Gabor() {
        rfType = RFtype::nonOppopent_gabor;
    }
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, bool strictStrength, Float sig = 1.177) override {
        this->n = n;
        this->sfreq = sfreq;
        this->phase = 0;
        this->amp = 0.5*amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}
    Float check_opponency(InputType iType, Float &modulation) override {
        Float opponent;
        OutputType currType;
		if (modulation < 0.5) {
			// switch to the dominant type
		    modulation = 1 - modulation;
			currType = static_cast<OutputType> ( 1- static_cast<Size>(oType) );
			assert(currType == OutputType::LonMon || currType == OutputType::LoffMoff);
		} else {
            currType = oType;
        }
		opponent = 1.0 * match_OnOff(iType, currType, modulation);
        return opponent;
    }

    Float get_prob(Float opponent, Float modulation, Float envelope) override {
        return envelope * (0.5 + amp * opponent * modulation);
    }
};

struct NonOpponent_CS: LinearReceptiveField {
	// center-surround
    NonOpponent_CS() {
        rfType = RFtype::nonOppopent_cs;
    }
    void setup_param(Size n, Float sfreq, Float phase, Float amp, Float theta, Float a, Float baRatio, OutputType oType, bool strictStrength, Float sig = 1.177) override {
        this->n = n;
        this->sfreq = 0.0;
        this->phase = 0.0;
        this->amp = 0.5*amp;
        this->theta = theta;
        this->a = a;
        this->baRatio = baRatio;
        this->oType = oType;
        this->sig = sig;
	}

    Float modulate(Float x, Float y) override {
        return 1.0;
    }

    Float check_opponency(InputType iType, Float &modulation) override {
		return 1.0 * match_OnOff(iType, oType, modulation);
	}

    Float get_prob(Float opponent, Float modulation, Float envelope) override {
        return envelope * (0.5 + amp * opponent * modulation);
    }
};

struct load_prop {

};

#endif
