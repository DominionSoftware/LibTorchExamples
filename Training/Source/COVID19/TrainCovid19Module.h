#ifndef TRAIN_COVID19_MODULE_
#define TRAIN_COVID19_MODULE_

#include <memory>
#include "../Common/IDataSet.h"
#include "COVID19DataSet.h"

namespace torch_explorer
{
    class Covid19Module;
    
    // Function to train a Covid19Module with the IDataSet interface
    void TrainCovid19Module(
        std::shared_ptr<Covid19Module> model, 
        std::shared_ptr<IDataSet<Covid19>> trainData, 
        std::shared_ptr<IDataSet<Covid19>> testData,
        size_t num_epochs,
        double learningRate = 0.001, 
        size_t logInterval = 100);
}

#endif // TRAIN_COVID19_MODULE_
