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
            // TODO: skapa ett nytt MLContext och som du döper till context och sätt seed värdet till 0


            // Vi laddar in CSV filen och spearerar på komma tecknet
            // TODO: skapa en variabel som heter data och ladda in från contextet, från CSV filen du använder metoden LoadFromTextFile för att ladda in texten sen separera med komma tecken. 


            // Vi skapar pipelinen och berättar för modellen att vi vill klassificera värdet "PovertyRate"
            var pipeline = context.Transforms.NormalizeMinMax("PovertyRate")
                .Append(context.Transforms.Concatenate("Features", "PovertyRate"))
                .Append(context.Regression.Trainers.Ols());

            //Med Fit funktionen startar vi själva träningen!
            //TODO: Starta träna modellen genom att skapa en variabel som heter model sen anropa Fit metoden från pipelinen och passa in data variabeln.
            var model = pipeline.Fit(data);

            // Vi använder modellen för att göra predektion
            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);
            var input = new Input { PovertyRate = 19.7f };
            var prediction = predictor.Predict(input);

            // Vi printar ut vad vår estimerade birth rate är samt visar vilken birth rate som faktist stämmer från "facit"
            //TODO: Skapa en till WriteLine output där du tar ut värdet BirthRate från prediction variabeln.
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