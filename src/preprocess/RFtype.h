#ifndef RFTYPE_H
#define RFTYPE_H
#include <vector>
#include "types.h"
#include "util.h"

enum class RFtype {
	nonOppopent,  // different cone input correlates, i.e., have the same sign, in center surround, find in V1
	nonOppopent_concentric,  // different cone input correlates, i.e., have the same sign, in center surround, find in V1
	singleOppopent, // center surround cone opponency, LGN and V1
	singleOppopent_mixed, // mixed surround, i.e., center cone has spatial opponency (diff. of Gaussian), LGN and V1
	doubleOppopent, // center surround cone and spatial opponency, V1
    doubleOppopent_gabor // cone and spatial opponency in Gabor profile, V1
};

enum class OutputType { // V1 local RF 
    // if double peak, choose the first on the left
    // surround ignored
    // non-opponent
    LonMon, 
    LoffMoff, 
    // opponent
    LonMoff,
    LoffMon 
}

enum class InputType { // LGN
    // center-surround
    LonMoff,
    LoffMon,
    MonLoff,
    MoffLon
    // ignore mixed surround, almost always from a different cone type
};

// normalize x to [-1,1], y to baRatio*[-1,1]
auto transform_coord_to_unitRF(Float x, Float y, const Float mx, const Float my, const Float theta, const Float a) {
    // a is half-width at the x-axis
    x = (x - mx)/a;
    y = (y - my)/a;
    Float new_x, new_y;
    new_x = cos(theta) * x + sin(theta) * y;
	new_y = -sin(theta) * x + cos(theta) * y;
    return make_pair(new_x, new_y);
}

struct LinearReceptiveField { // RF sample without implementation of check_opponency
	vector<Float> prob;
    Size n;
    Float sfreq, phase, amp, theta, radius, baRatio, sig;
    OutputType oType;
    // default constructor
    LinearReceptiveField(Size n, Float sfreq, Float phase, Float amp, Float theta, Float radius, Float baRatio, OutputType oType, Float sig = 1.177):
        n(n),
        sfreq(sfreq),
        phase(phase),
        amp(amp),
        theta(theta),
        radius(radius),
        baRatio(baRatio),
        oType(oType),
        sig(sig)
    {}
 
    virtual Size construct_RF(vector<Float> &x, vector<Float> &y, vector<Float> &iLM_OF, vector<Size> &idList, vector<Float> strengthList, RandomEngine &rGen) {
        Size nConnected;
        if (n > 0) {
		    prob.reserve(n);
            // putative RF center
            tie(cx, cy) = average(x, y);
		    for (Size i = 0; i < n; i++) {
                Float norm_x, norm_y;
                // orient and normalize LGN coord
		        tie(norm_x, norm_y) = transform_coord_to_unitRF(x[i], y[i], cx, cy, theta, radius);
                // calc. prob. dist. distance-dependent envelope at coord.
                Float envelope = get_envelope(norm_x, norm_y);
                // calc. modulation of prob at coord.
                Float modulation = modulate(norm_x, norm_y);
                // calc. cone and position opponency at coord.
	            Float opponent = check_opponency(iLM_OF[i], oLM_OF, modulation);
                prob.push_back(get_prob(opponent, modulation, envelope, amp));
            }
            // make connection and update ID and strength list
            nConnected = connect(idList, strengthList, rGen);
        }  else {
            nConnected = 0;
        }
        return nConnected;
    }
    // Probability envelope based on distance
    virtual Float get_envelope(Float x, Float y, Float amp, Float baRatio, Float sig = 1.1775) {
        return exp(-0.5*(pow(x/sig,2)+pow(y/(sig*baRatio),2)));
        // baRatio comes from Dow et al., 1981
    }
    // Full cosine modulation
    virtual Float modulate(Float x, Float y) {
        return 1.0 + cos(sfreq * x * M_PI + phase);
        // sfreq should be given as a percentage of the width
        // when sfreq == 1, the RF contain a full cycle of cos(x), x\in[-pi, pi]
    }
    // To be implemented by derived class
    virtual Float check_opponency(InputType iType, OutputType oType, Float modulation) = 0;
    // produce connection prob. at coord. for i-th LGN
    virtual Float get_prob(bool opponent, Float modulation, Float envelope, Float amp) {
        return envelope * (1.0 + amp * opponent * modulation);
    }
    // normalize prob.
    void normalize(Float percent) {
	    // average connection probability is controlled at percent.
        const Float norm = accumulate(prob.begin(), prob.end(), 0.0) / (percent * prob.size());
	    //cout << "norm = " << norm << "\n";
	    //print_list<Float>(prob);
	    assert(!isnan(norm));
        //Float sum = 0.0;
        for (Size i=0; i<prob.size(); i++) {
            prob[i] = prob[i] / norm;
            //sum += prob[i];
        }
    //cout << percent*prob.size() << " ~ " << sum << "\n";
    }
    // make connections
    virtual Size connect(vector<Size> &idList, vector<Float> strengthList, RandomEngine &rGen) {
		// make connections and normalized strength i.e., if prob > 1 then s = 1 else s = prob
        uniform_real_distribution<Float> uniform(0,1);
		strengthList.reserve(n);
		vector<Int> newList;
		newList.reserve(n);
		for (Size i = 0; i < n; i++) {
			if (uniform(rGen) < prob[i]) {
				newList.push_back(idList[i]);
				if (prob[i] > 1) {
					strengthList.push_back(prob[i]);
				} else {
					strengthList.push_back(1);
				}
			}
		}
		idList = newList;
		idList.shrink_to_fit();
        return idList.size();
    }
}

struct NonOpponent: LinearReceptiveField {
    NonOpponentConcentric(Size n, Float amp, Float theta, Float radius, Float baRatio, OutputType oType, Float sig = 1.177):
        n(n),
        sfreq(sfreq),
        phase(phase),
        amp(amp),
        theta(theta),
        radius(radius) 
        baRatio(baRatio),
        oType(oType),
        sig(sig)
    {}

    Float modulate(Float x, Float y) {
        return 1.0;
        // sfreq should be given as a percentage of the width
        // when sfreq == 1, the RF contain a full cycle of cos(x), x\in[-pi, pi]
    }

    Float check_opponency(InputType iType, OutputType oType, Float modulation) override {
        Int LM, OF;
        Float opponent;
        // no opponency in cone, only in space
	    if (iLM_OF % 2 == oLM_OF % 2) {
	    	opponent = 1;
	    } else {
	    	opponent = -1;
	    }
        return opponent;
    }

    Float get_prob(bool opponent, Float modulation, Float envelope, Float amp) {
        return envelope * (1.0 + amp * opponent * modulation);
    }
}

struct NonOpponentConcentric: LinearReceptiveField {
    NonOpponentConcentric(Size n, Float amp, Float theta, Float radius, Float baRatio, OutputType oType, Float sig = 1.177):
        n(n),
        sfreq(0), // no modulation inside the envelope
        phase(0),
        amp(1),
        theta(theta),
        radius(radius) 
        baRatio(baRatio),
        oType(oType),
        sig(sig)
    {}

    Float modulate(Float x, Float y) override {
        return 0.5;
    }

    Float check_opponency(InputType iType, OutputType oType, Float modulation) override {
        Float opponent;
        switch (oType) {
            case OutputType::LonMon:
                if (iType == InputType::LonMoff || iType == InputType::MonLoff) {
                    opponent = 1;
                } else {
                    opponent = -1;
                }
                break;
            case OutputType::LoffMoff:
                if (iType == InputType::LoffMon || iType == InputType::MoffLon) {
                    opponent = 1;
                } else {
                    opponent = -1;
                }
            default:
                throw("There's no implementation of such combination of cone types for non-opponent concentric RF");
        }
        return opponent;
    }

    Float get_prob(bool opponent, Float modulation, Float envelope, Float amp) override {
        return envelope * (0.5 + amp * opponent * modulation);
    }
}
};

#endif
