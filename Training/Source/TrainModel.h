
#include "IDataSet.h"
#ifndef TRAIN_MODEL_

#include <memory>
void TrainModel(std::shared_ptr<IDataSet> trainData, std::shared_ptr<IDataSet> testData, size_t num_epochs,
	double learningRate = 0.0009, size_t logInterval = 100);

#endif