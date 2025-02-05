// BetaDistribution.h
#ifndef BETA_DISTRIBUTION_H
#define BETA_DISTRIBUTION_H

#include <random>

namespace torch_explorer {

    class BetaDistribution {
    public:
        BetaDistribution()
            : BetaDistribution(1.0f, 1.0f)

        {
        }

        BetaDistribution(float alpha, float beta)
            : alpha_(alpha), beta_(beta),
            gamma_alpha_(alpha, 1.0),
            gamma_beta_(beta, 1.0) {
        }

        template<typename Generator>
        float operator()(Generator& gen) {
            float x = gamma_alpha_(gen);
            float y = gamma_beta_(gen);
            return x / (x + y);
        }

    private:
        float alpha_;
        float beta_;
        std::gamma_distribution<float> gamma_alpha_;
        std::gamma_distribution<float> gamma_beta_;
    };

} // namespace torch_explorer

#endif
