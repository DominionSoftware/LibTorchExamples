
#ifndef TRAIN_MODEL_
#include <memory>
#include "IDataSet.h"

void TrainModel(std::shared_ptr<IDataSet> trainData, std::shared_ptr<IDataSet> testData, size_t num_epochs,
	double learningRate = 0.001, size_t logInterval = 100);

#endif