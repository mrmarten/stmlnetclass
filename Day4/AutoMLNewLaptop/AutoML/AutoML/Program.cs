using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;

namespace AutoML
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\dataframe-laptops.csv";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Ladda in datan
            var data = context.Data.LoadFromTextFile<Input>(_path, hasHeader: true, separatorChar: ',');

            // Skapa ett regressions experiment
            var settings = new RegressionExperimentSettings
            {
                MaxExperimentTimeInSeconds = 30, // körtid i sekunder 30 minuters total maximaltid.
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
                brand = "HP",
                display_size = 14.0f,
                processor_type = "Intel Core i5-8250U",
                graphics_card = "Intel UHD Graphics 620",
                disk_space = "1 TB HDD",
                discount_price = 0.0f

            };

            var prediction = predictor.Predict(input);

            // Vi får svar om det estimerade priset jämförelsevis med vad faktiskt objektet sålde för i verkligheten.

            Console.WriteLine($"Predicted price: ${prediction.Price:n0}; Actual price: $3799,000");
            Console.WriteLine();
        }
    }

    public class Input
    {
        [LoadColumn(1)]
        public string brand;

        [LoadColumn(3)]
        public float display_size;

        [LoadColumn(4)]
        public string processor_type;

        [LoadColumn(5)]
        public string graphics_card;

        [LoadColumn(6)]
        public string disk_space;


        [LoadColumn(7), ColumnName("Label")]
        public float discount_price;


    }

    public class Output
    {
        [ColumnName("Score")]
        public float Price;
    }
}
