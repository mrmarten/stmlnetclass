using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace SimpleRegression
{
    class Program
    {
        // Vi läser in CSV filen
        static readonly string _path = "..\\..\\..\\Data\\poverty.csv";

        static void Main(string[] args)
        {
            // Vi skapar en MLContext
            var context = new MLContext(seed: 0);

            // Vi laddar in CSV filen och spearerar på komma tecknet
            var data = context.Data.LoadFromTextFile<Input>(_path, hasHeader: true, separatorChar: ',');

            // Vi skapar pipelinen och berättar för modellen att vi vill klassificera värdet "PovertyRate"
            var pipeline = context.Transforms.NormalizeMinMax("PovertyRate")
                .Append(context.Transforms.Concatenate("Features", "PovertyRate"))
                .Append(context.Regression.Trainers.Ols());

            //Med Fit funktionen startar vi själva träningen!
            var model = pipeline.Fit(data);

            // Vi använder modellen för att göra predektion
            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);
            var input = new Input { PovertyRate = 19.7f };
            var prediction = predictor.Predict(input);

            // Vi printar ut vad vår estimerade birth rate är samt visar vilken birth rate som faktist stämmer från "facit"
            Console.WriteLine($"Predicted birth rate: {prediction.BirthRate:0.##}");
            Console.WriteLine($"Actual birth rate: 58.10");
            Console.WriteLine();
        }
    }

    public class Input
    {
        [LoadColumn(1)]
        public float PovertyRate;

        [LoadColumn(5), ColumnName("Label")]
        public float BirthRate;
    }

    public class Output
    {
        [ColumnName("Score")]
        public float BirthRate;
    }
}