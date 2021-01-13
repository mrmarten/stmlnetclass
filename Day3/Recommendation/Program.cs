using System;

using recommendation.ML;

namespace recommendation
{
    class Program
    {
        static void Main(string[] args)
        {
            // Efter du har kompilerat denna solution kan du testa exempel genom att exekvera exe filen i bin katalog och använda input.json filen som parameter för texta ett exempel.
            // C:\dev\stmlnetclass\Day3\Recommendation\bin\Debug\netcoreapp3.0>recommendation.exe predict input.json

            if (args.Length < 2)
            {
                Console.WriteLine($"Invalid arguments passed in, exiting.{Environment.NewLine}{Environment.NewLine}Usage:{Environment.NewLine}" +
                                  $"predict <path to input file>{Environment.NewLine}" +
                                  $"or {Environment.NewLine}" +
                                  $"train <path to training data file> <path to test data file>{Environment.NewLine}" +
                                  $"or {Environment.NewLine}" + $"extract <path to training folder> <path to test folder>{Environment.NewLine}");

                return;
            }

            switch (args[0])
            {
                case "predict":
                    new Predictor().Predict(args[1]);
                    break;
                case "train":
                    new Trainer().Train(args[1], args[2]);
                    break;
                default:
                    Console.WriteLine($"{args[0]} is an invalid option");
                    break;
            }
        }
    }
}