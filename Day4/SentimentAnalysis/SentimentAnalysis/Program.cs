﻿using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\yelp_labelled.tsv";

        static readonly string[] _samples =
        {
            "The food was great and the service was excellent",
            "The food was bland and the service was average",
            "The food was bland and the service was below average",
            "The only thing worse than the food was the terrible service",
            "I wouldn't let my dog eat here"
        };

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Ladda datan
            var data = context.Data.LoadFromTextFile<Input>(_path, hasHeader: false);

            // Splittra datan till ett träning och test data. Testa med 20 procent utav det to.
            var trainTestData = context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // Skapa en pipeline och sätt Sentiment texten som vi vill ha input som Feature.
            var pipeline = context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "SentimentText")
                .Append(context.BinaryClassification.Trainers.FastTree(numberOfLeaves: 50, minimumExampleCountPerLeaf: 20));

            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainData);

            // Evaluera modellen och skapa mätvärden
            var predictions = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"F1: {metrics.F1Score:P2}");
            Console.WriteLine();

            // Utvärdera modellen genom cross-validering
            var scores = context.BinaryClassification.CrossValidate(data, pipeline, numberOfFolds: 5);
            var mean = scores.Average(x => x.Metrics.F1Score);
            Console.WriteLine($"Mean cross-validated F1 score: {mean:P2}");

            // Använd modellen för att göra flera prediktioner
            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);

            foreach (var sample in _samples)
            {
                var input = new Input { SentimentText = sample };
                var prediction = predictor.Predict(input);

                Console.WriteLine();
                Console.WriteLine($"{input.SentimentText}");
                Console.WriteLine($"Sentiment score: {prediction.Probability}");
                Console.WriteLine($"Sentiment: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")}");
            }

            Console.WriteLine();
        }
    }

    public class Input
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    public class Output
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}