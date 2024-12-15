#include <iostream>
#include <array>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>

class SNN {
private:
    static constexpr size_t NUMINPUTNODES   = 2;
    static constexpr size_t NUMHIDDENNODES  = 2;
    static constexpr size_t NUMOUTPUTNODES  = 1;
    static constexpr size_t NUMTRAININGSETS = 4;

    static constexpr double lr = 5;

    std::array<double, NUMHIDDENNODES> hiddenNodes;
    std::array<double, NUMOUTPUTNODES> outputNodes;

    std::array<double, NUMHIDDENNODES> ActivationHiddenNodes;
    std::array<double, NUMOUTPUTNODES> ActivationOutputNodes;

    std::array<std::array<double, NUMHIDDENNODES>, NUMINPUTNODES>  hiddenWeights;
    std::array<std::array<double, NUMOUTPUTNODES>, NUMHIDDENNODES> outputWeights;

    std::array<double, NUMHIDDENNODES> hiddenBias;
    std::array<double, NUMOUTPUTNODES> outputBias;

    std::array<std::array<double, NUMINPUTNODES>, NUMTRAININGSETS> trainInput = {{
        {{0.0, 0.0}},
        {{1.0, 0.0}},
        {{0.0, 1.0}},
        {{1.0, 1.0}}
    }};

    std::array<std::array<double, NUMOUTPUTNODES>, NUMTRAININGSETS> trainOutput = {{
        {{0.0}},
        {{1.0}},
        {{1.0}},
        {{0.0}}
    }};

    void InitWeightsAndBiases();
    void ShuffleTest();
    double Sigmoid(double x);
    double DSigmoid(double x);
    double Cost(double actualOutput, double expectedOutput);
    double DCost(double actualOutput, double expectedOutput);

public: 
    SNN();
    void Train(size_t numEpochs);
    void Test();

};

SNN::SNN() {
    InitWeightsAndBiases();
}

double SNN::Sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double SNN::DSigmoid(double x) {
    double activation = Sigmoid(x);
    return activation * (1 - activation);
}

double SNN::Cost(double actualOutput, double expectedOutput) {
    return (actualOutput - expectedOutput) * (actualOutput - expectedOutput);
}

double SNN::DCost(double actualOutput, double expectedOutput) {
    return 2 * (actualOutput - expectedOutput);
}

void SNN::InitWeightsAndBiases() {
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> distribution(0, 1);
    
    for(auto &i : hiddenWeights) {
        for(auto &v : i) {
            v = distribution(generator);
        }
    }

    for(auto &i : outputWeights) {
        for(auto &v : i) {
            v = distribution(generator);
        }
    }

    for(auto &v : hiddenBias) {
        v = distribution(generator);
    }

    for(auto &v : outputBias) {
        v = distribution(generator);
    }
}

void SNN::ShuffleTest() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    std::shuffle(trainInput.begin(), trainInput.end(), std::default_random_engine(seed));
    std::shuffle(trainOutput.begin(), trainOutput.end(), std::default_random_engine(seed));
}

void SNN::Train(size_t numEpochs) {
    for(size_t epoch = 0; epoch < numEpochs; epoch++) {
        ShuffleTest();

        for(size_t setNo = 0; setNo < NUMTRAININGSETS; setNo++) {
            
            // forward pass input --> hidden
            for(size_t hiddenNo = 0; hiddenNo < NUMHIDDENNODES; hiddenNo++) {
                hiddenNodes[hiddenNo] = hiddenBias[hiddenNo];
                
                for(size_t inputNo = 0; inputNo < NUMINPUTNODES; inputNo++) {
                    hiddenNodes[hiddenNo] += trainInput[setNo][inputNo] * hiddenWeights[inputNo][hiddenNo];
                }

                ActivationHiddenNodes[hiddenNo] = Sigmoid(hiddenNodes[hiddenNo]);
            }

            // forward pass hidden --> output
            for(size_t outputNo = 0; outputNo < NUMOUTPUTNODES; outputNo++) {
                outputNodes[outputNo] = outputBias[outputNo];
                
                for(size_t hiddenNo = 0; hiddenNo < NUMHIDDENNODES; hiddenNo++) {
                    outputNodes[outputNo] += ActivationHiddenNodes[hiddenNo] * outputWeights[hiddenNo][outputNo];
                }

                ActivationOutputNodes[outputNo] = Sigmoid(outputNodes[outputNo]);
            }

            std::cout << "INPUT: "           << trainInput[setNo][0]     << " " << trainInput[setNo][1] << " " \
                      << "EXPECTED OUTPUT: " << trainOutput[setNo][0]    << " " \
                      << "ACTUAL OUTPUT: "   << std::fixed << std::setprecision(5) << ActivationOutputNodes[0] << " " \
                      << "COST: " << std::fixed << std::setprecision(10) << Cost(ActivationOutputNodes[0], trainOutput[setNo][0]) \
                      << std::noshowpoint << std::setprecision(0) << std::endl;

            // backpropagation

            std::array<double, NUMOUTPUTNODES> deltaOutput;
            std::array<double, NUMHIDDENNODES> deltaHidden;

            // populate deltaOutput
            for(size_t outputNo = 0; outputNo < NUMOUTPUTNODES; outputNo++) {
                deltaOutput[outputNo] = DCost(ActivationOutputNodes[outputNo], trainOutput[setNo][outputNo]) * DSigmoid(outputNodes[outputNo]);
            }

            // populate deltaHidden
            for(size_t hiddenNo = 0; hiddenNo < NUMHIDDENNODES; hiddenNo++) {
                deltaHidden[hiddenNo] = 0;

                for(size_t outputNo = 0; outputNo < NUMOUTPUTNODES; outputNo++) {
                    deltaHidden[hiddenNo] += deltaOutput[outputNo] * outputWeights[hiddenNo][outputNo];
                }

                deltaHidden[hiddenNo] = deltaHidden[hiddenNo] * DSigmoid(hiddenNodes[hiddenNo]);
            }

            // update output weights and bias
            for(size_t outputNo = 0; outputNo < NUMOUTPUTNODES; outputNo++) {
                outputBias[outputNo] -= deltaOutput[outputNo] * lr;

                for(size_t hiddenNo = 0; hiddenNo < NUMHIDDENNODES; hiddenNo++) {
                    outputWeights[hiddenNo][outputNo] -= deltaOutput[outputNo] * ActivationHiddenNodes[hiddenNo] * lr;
                }
            }

            // update hidden weights and bias
            for(size_t hiddenNo = 0; hiddenNo < NUMHIDDENNODES; hiddenNo++) {
                hiddenBias[hiddenNo] -= deltaHidden[hiddenNo] * lr;

                for(size_t inputNo = 0; inputNo < NUMINPUTNODES; inputNo++) {
                    hiddenWeights[inputNo][hiddenNo] -= deltaHidden[hiddenNo] * trainInput[setNo][inputNo] * lr;
                }
            }
        }
    }
}

void SNN::Test() {
    ShuffleTest();

    std::cout << "\nTEST:\n";

    for(size_t setNo = 0; setNo < NUMTRAININGSETS; setNo++) {
        
        // forward pass input --> hidden
        for(size_t hiddenNo = 0; hiddenNo < NUMHIDDENNODES; hiddenNo++) {
            hiddenNodes[hiddenNo] = hiddenBias[hiddenNo];
            
            for(size_t inputNo = 0; inputNo < NUMINPUTNODES; inputNo++) {
                hiddenNodes[hiddenNo] += trainInput[setNo][inputNo] * hiddenWeights[inputNo][hiddenNo];
            }

            ActivationHiddenNodes[hiddenNo] = Sigmoid(hiddenNodes[hiddenNo]);
        }

        // forward pass hidden --> output
        for(size_t outputNo = 0; outputNo < NUMOUTPUTNODES; outputNo++) {
            outputNodes[outputNo] = outputBias[outputNo];
            
            for(size_t hiddenNo = 0; hiddenNo < NUMHIDDENNODES; hiddenNo++) {
                outputNodes[outputNo] += ActivationHiddenNodes[hiddenNo] * outputWeights[hiddenNo][outputNo];
            }

            ActivationOutputNodes[outputNo] = Sigmoid(outputNodes[outputNo]);
        }
        
        std::cout << "INPUT: "           << trainInput[setNo][0]     << " " << trainInput[setNo][1] << " " \
                  << "EXPECTED OUTPUT: " << trainOutput[setNo][0]    << " " \
                  << "ACTUAL OUTPUT: "   << round(ActivationOutputNodes[0]) << std::endl;
    }
}

int main() {
    SNN network;
    network.Train(10000);
    network.Test();
}