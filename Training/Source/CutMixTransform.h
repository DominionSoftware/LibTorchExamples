// CutMixTransform.h
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
            // Skip CutMix with probability (1 - prob)
            if (std::uniform_real_distribution<>(0, 1)(gen_) > prob_)
            {
                return example1;
            }

            auto image1 = example1.data;
            auto label1 = example1.target;
            auto image2 = example2.data;
            auto label2 = example2.target;

            // Generate random box parameters using beta distribution
            float lambda = beta_dist_(gen_);
            int H = image1.size(1); // Height (32 for CIFAR100)
            int W = image2.size(2); // Width (32 for CIFAR100)

            // Calculate cut size
            int cut_w = static_cast<int>(std::sqrt(1.0 - lambda) * W);
            int cut_h = static_cast<int>(std::sqrt(1.0 - lambda) * H);

            // Generate random center point
            int cx = std::uniform_int_distribution<>(0, W)(gen_);
            int cy = std::uniform_int_distribution<>(0, H)(gen_);

            // Calculate box coordinates
            int x1 = std::max(cx - cut_w / 2, 0);
            int y1 = std::max(cy - cut_h / 2, 0);
            int x2 = std::min(cx + cut_w / 2, W);
            int y2 = std::min(cy + cut_h / 2, H);

            // Create mixed image
            auto mixed_image = image1.clone();
            mixed_image.slice(1, y1, y2).slice(2, x1, x2) =
                image2.slice(1, y1, y2).slice(2, x1, x2);

            // Calculate mixing ratio and use majority label
            float mix_ratio = 1.0f - static_cast<float>((x2 - x1) * (y2 - y1)) / (W * H);
            auto mixed_label = (mix_ratio > 0.5) ? label1 : label2;

            return { mixed_image, mixed_label };
        }

    private:
        float alpha_;
        float prob_;
        BetaDistribution beta_dist_;
        std::mt19937 gen_;
    };
} // namespace torch_explorer

#endif