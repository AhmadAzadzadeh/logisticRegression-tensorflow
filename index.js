require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const _ = require("lodash");
const LogisticRegression = require("./logistic-regression");
const mnist = require("mnist-data");

const loadData = () => {
    const mnistData = mnist.training(0, 60000);

    const features = mnistData.images.values.map(image => _.flatMap(image));
    const encodedLabels = mnistData.labels.values.map(label => {
        const row = new Array(10).fill(0);
        row[label] = 1;
        return row;
    });

    return { features, encodedLabels };
};

const { features, encodedLabels } = loadData();

const regression = new LogisticRegression(features, encodedLabels, {
    learningRate: 1,
    iterations: 80,
    batchSize: 500
});

regression.train();

const testMnistData = mnist.testing(0, 10000);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
});

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log(accuracy);