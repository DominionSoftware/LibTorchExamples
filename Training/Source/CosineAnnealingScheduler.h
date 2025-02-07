#ifndef COSINE_ANNEALING_SCHEDULER_
#define COSINE_ANNEALING_SCHEDULER_
#define _USE_MATH_DEFINES
#include "ISchedulerStep.h"
#include <torch/optim/schedulers/lr_scheduler.h>
#include <cmath>
#include "IOptimizer.h"


namespace torch_explorer 
{

class CosineAnnealingScheduler : public ISchedulerStep 
{
public:
    CosineAnnealingScheduler(IOptimizer& optimizer,
                            size_t T_max,
                            double eta_min = 0,
                            size_t T_mult = 2)
        : optimizer_(optimizer)
        , T_max_(T_max)
        , eta_min_(eta_min)
        , T_mult_(T_mult)
        , T_cur_(0)
        , cycle_(0)
        , initial_lr_(optimizer.current().param_groups()[0].options().get_lr())
    {
    }

    void doStep(double /* param */ = 0) override;
   

    std::vector<double> getLearningRates()
	{
        std::vector<double> current_lrs;
        for (const auto& group : optimizer_.current().param_groups()) 
        {
            current_lrs.push_back(group.options().get_lr());
        }
        return current_lrs;
    }
    void resetScheduler(double new_initial_lr = -1)
	{
        T_cur_ = 0;
        cycle_ = 0;
        if (new_initial_lr > 0) 
        {
            initial_lr_ = new_initial_lr;
        }
    }


    CosineAnnealingScheduler(const CosineAnnealingScheduler&) = delete;
    CosineAnnealingScheduler& operator=(const CosineAnnealingScheduler&) = delete;

   


protected:
   

private:
    IOptimizer& optimizer_;
    size_t T_max_;      // Period before first restart
    double eta_min_;    // Minimum learning rate
    size_t T_mult_;     // Factor to increase T_max after each restart
    size_t T_cur_;      // Current number of steps in cycle
    size_t cycle_;      // Current cycle number
    double initial_lr_;

};

} 

#endif