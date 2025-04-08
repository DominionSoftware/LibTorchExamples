#ifndef TRANSFORMS_
#define TRANSFORMS_
#include <random>


namespace torch_exporer
{

    class RandomHorizontalFlip {
  

    public:
        RandomHorizontalFlip(double p = 0.5) : probability_(p), distribution_(0.0, 1.0) 
        {
            std::random_device rd;
            generator_ = std::mt19937(rd());
        }

        torch::Tensor operator()(torch::Tensor tensor) 
        {
            if (distribution_(generator_) < probability_)
            {
                return tensor.flip(2); // Flip along width dimension  
            }
            return tensor;
        }


    private:
        double probability_;
        std::mt19937 generator_;
        std::uniform_real_distribution<double> distribution_;
    };


    class ComposeTransforms 
    {
   

    public:
        ComposeTransforms(const std::vector<std::function<torch::Tensor(torch::Tensor)>>& transforms_list)
            : transforms_(transforms_list) {}

        torch::Tensor operator()(torch::Tensor tensor) 
        {
            for (auto& transform : transforms_)
            {
                tensor = transform(tensor);
            }
            return tensor;
        }

    private:
        std::vector<std::function<torch::Tensor(torch::Tensor)>> transforms_;
    };

    class RandomRotation {
    private:
        double degrees;
        std::mt19937 generator;
        std::uniform_real_distribution<double> distribution;

    public:
        RandomRotation(double degrees_max = 10.0)
            : degrees(degrees_max),
            distribution(-degrees, degrees) {
            std::random_device rd;
            generator = std::mt19937(rd());
        }

        torch::Tensor operator()(torch::Tensor tensor) {
            // Check dimensionality - handle both 3D (CHW) and 4D (NCHW) tensors
            bool is_batch = tensor.dim() == 4;

            // If single image (CHW), add batch dimension temporarily
            if (!is_batch) {
                tensor = tensor.unsqueeze(0);  // Add batch dimension
            }

            double angle = distribution(generator);
            double rad = angle * M_PI / 180.0;

            // Create rotation matrix
            float cos_val = std::cos(rad);
            float sin_val = std::sin(rad);

            // Create 2D rotation matrix (theta) for affine_grid
            auto theta = torch::zeros({ tensor.size(0), 2, 3 }, tensor.options());
            theta.index_put_({ torch::indexing::Slice(), 0, 0 }, cos_val);
            theta.index_put_({ torch::indexing::Slice(), 0, 1 }, -sin_val);
            theta.index_put_({ torch::indexing::Slice(), 1, 0 }, sin_val);
            theta.index_put_({ torch::indexing::Slice(), 1, 1 }, cos_val);

            // Create the sampling grid
            auto grid = torch::nn::functional::affine_grid(
                theta,
                tensor.sizes(),  // Use the full size tensor
                false  // align_corners = false
            );

            // Apply grid to get rotated image
            auto rotated = torch::nn::functional::grid_sample(
                tensor,
                grid,
                torch::nn::functional::GridSampleFuncOptions().align_corners(false).mode(torch::kBilinear)
            );

            // If we added a batch dimension, remove it
            if (!is_batch) {
                rotated = rotated.squeeze(0);
            }

            return rotated;
        }
    };

    class RandomBrightness 
    {

    public:
        RandomBrightness(double range = 0.2)
            : factor_range_(range),
            distribution_(1.0 - range, 1.0 + range) 
        {
            std::random_device rd;
            generator_ = std::mt19937(rd());
        }

        torch::Tensor operator()(torch::Tensor tensor) 
        {
            double factor = distribution_(generator_);
            return torch::clamp(tensor * factor, 0.0, 1.0);
        }

    private:
        double factor_range_;
        std::mt19937 generator_;
        std::uniform_real_distribution<double> distribution_;
    };

}



#endif