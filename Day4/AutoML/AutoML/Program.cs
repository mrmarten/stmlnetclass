using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;

namespace AutoML
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\pacific-heights.csv";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Ladda in datan
            var data = context.Data.LoadFromTextFile<Input>(_path, hasHeader: true, separatorChar: ',');

            // Skapa ett regressions experiment
            var settings = new RegressionExperimentSettings
            {
                MaxExperimentTimeInSeconds = 1800, // körtid i sekunder 30 minuters total maximaltid.
                OptimizingMetric = RegressionMetric.RSquared,
                CacheDirectory = null
            };

            // Här berättar vi att vi vill köra AutoML som experiment, men kolla vilka andra alternativ som finns att välja än CreateRegressionExperiment.
            var experiment = context.Auto().CreateRegressionExperiment(settings);

            // Starta experimentet
            Console.WriteLine("Running the experiment...");
            var result = experiment.Execute(data);

            // Vi berättar att vi vill ha tillbaka mätvärden från den bästa modellen som AutoML kunde hitta från körningen.
            RegressionMetrics metrics = result.BestRun.ValidationMetrics;
            Console.WriteLine($"R2 score: {metrics.RSquared:0.##}");
            Console.WriteLine();

            // Använd den bästa modellen för att göra en prediktion
            var predictor = context.Model.CreatePredictionEngine<Input, Output>(result.BestRun.Model);

            // Vi ger modellen ett test från ett exempel som den inte tränat på innan.
            var input = new Input
            {
                Bathrooms = 1.0f,
                Bedrooms = 1.0f,
                TotalRooms = 3.0f,
                FinishedSquareFeet = 653.0f,
                UseCode = "Condominium",
                LastSoldPrice = 0.0f
            };

            var prediction = predictor.Predict(input);

            // Vi får svar om det estimerade priset jämförelsevis med vad faktiskt objektet sålde för i verkligheten.

            Console.WriteLine($"Predicted price: ${prediction.Price:n0}; Actual price: $665,000");
            Console.WriteLine();
        }
    }

    public class Input
    {
        [LoadColumn(1)]
        public float Bathrooms;

        [LoadColumn(2)]
        public float Bedrooms;

        [LoadColumn(3)]
        public float FinishedSquareFeet;

        [LoadColumn(5), ColumnName("Label")]
        public float LastSoldPrice;

        [LoadColumn(9)]
        public float TotalRooms;

        [LoadColumn(10)]
        public string UseCode;
    }

    public class Output
    {
        [ColumnName("Score")]
        public float Price;
    }
}
