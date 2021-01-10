using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace SpamClassifier
{
    class Program
    {
        // Vi laddar in redan markerade CSV filen med både SPAM mail och ICKE-SPAM mail.
        static readonly string _path = "..\\..\\..\\Data\\ham-spam.csv";

        // Dessa exempel använder vi senare för vår tränade modell att klassificera om meningen innehåller SPAM eller inte.
        static readonly string[] _samples =
        {
            "If you can get the new revenue projections to me by Friday, I'll fold them into the forecast.",
            "Can you attend a meeting in Atlanta on the 16th? I'd like to get the team together to discuss in-person.",
            "Why pay more for expensive meds when you can order them online and save $$$?"
        };

        static void Main(string[] args)
        {
            // Vi sätter ett seed value till 0, så att vi kan träna om modellen med "samma data" och veta om vi gör framsteg eller bara slumpmässiga framsteg.
            var context = new MLContext(seed: 0);

            // Vi laddar in data och spearerar på komma tecknet
            var data = context.Data.LoadFromTextFile<Input>(_path, hasHeader: true, separatorChar: ',');

            // Vi testar sedan modellen med 20 procent utav resterande data från det 80 procentiga innehållet som vi använder för att träna modellen.
            // Samt sätter seed 0 så om vi gör ändringar utav ex, träningsmodellen att vi vet att vi gör framsteg eller inte.
            var trainTestData = context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // Vi skapar pipelinen vi sätter input columnen från text och döper om den till Features vi berättar sen att vi vill använda oss utav regressions modellen SDCA.
            var pipeline = context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Text")
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            // Sedan startar vi igång träningen för modellen
            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainData);

            // Vi utvärderar modellen och skriver ut en confusion matrix, vi ber att få mätvärden från modellen om värdet är satt att mailet är spam eller inte.
            var predictions = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

            // Här skriver vi ut flera olika mätvärden som är standard inom Machine Learning.

            Console.WriteLine();
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"F1: {metrics.F1Score:P2}");
            Console.WriteLine();

            // Nu använder vi modellen för att göra utvärderingar
            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);

            foreach (var sample in _samples)
            {
                var input = new Input { Text = sample };
                var prediction = predictor.Predict(input);

                Console.WriteLine();
                Console.WriteLine($"{input.Text}");
                Console.WriteLine($"Spam score: {prediction.Probability}");
                Console.WriteLine($"Classification: {(Convert.ToBoolean(prediction.Prediction) ? "Spam" : "Not spam")}");
            }

            Console.WriteLine();
        }
    }

    public class Input
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool IsSpam;

        [LoadColumn(1)]
        public string Text;
    }

    public class Output
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}