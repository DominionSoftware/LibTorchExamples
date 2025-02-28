#ifndef CUTMIX_TRANSFORM_H
#define CUTMIX_TRANSFORM_H

#include <torch/torch.h>
#include <random>
#include "BetaDistribution.h"

namespace torch_explorer
{
    class CutMixTransform
    {
    public:
        CutMixTransform(float alpha = 1.0, float prob = 0.5)
            : alpha_(alpha), prob_(prob), beta_dist_(alpha, alpha), gen_(std::random_device{}())
        {
        }

        torch::data::Example<> apply(const torch::data::Example<>& example1,
                                     const torch::data::Example<>& example2)
        {
            if (std::uniform_real_distribution<>(0, 1)(gen_) > prob_)
            {
                return example1;
            }

            auto image1 = example1.data;
            auto label1 = example1.target;
            auto image2 = example2.data;
            auto label2 = example2.target;

            float lambda = beta_dist_(gen_);
            int H = image1.size(1);
            int W = image2.size(2);

            int cut_w = static_cast<int>(std::sqrt(1.0 - lambda) * W);
            int cut_h = static_cast<int>(std::sqrt(1.0 - lambda) * H);

            int cx = std::uniform_int_distribution<>(cut_w / 2, W - cut_w / 2)(gen_);
            int cy = std::uniform_int_distribution<>(cut_h / 2, H - cut_h / 2)(gen_);

            int x1 = std::max(cx - cut_w / 2, 0);
            int y1 = std::max(cy - cut_h / 2, 0);
            int x2 = std::min(cx + cut_w / 2, W);
            int y2 = std::min(cy + cut_h / 2, H);

            auto mixed_image = image1.clone();
            mixed_image.slice(1, y1, y2).slice(2, x1, x2) = image2.slice(1, y1, y2).slice(2, x1, x2);

            auto mixed_label = lambda * label1 + (1 - lambda) * label2;

            return { mixed_image, mixed_label };
        }

    private:
        float alpha_;
        float prob_;
        BetaDistribution beta_dist_;
        std::mt19937 gen_;
    };
} 

#endif