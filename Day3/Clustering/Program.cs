using System;

using clustering.ML;

namespace clustering
{
    class Program
    {
        /*
          OBS: Efter vi har kompilerat applikationen kan vi köra träning och prediktionen med följande kommandon.
         
        Vi måste först träna modellen genom att passa sampledata.csv och testdata.csv filen.
         .\Clustering.exe train ..\..\..\Data\sampledata.csv ..\..\..\Data\testdata.csv

        För att sedan köra modellen så lägger vi simpelt till filnamnet för applikationen.
         .\Clustering.exe predict Clustering.exe
         
        
        Program class:en är där hela applikationen startas ifrån, vi tar emot två argument för tränings data och test data.
        */
        static void Main(string[] args)
        {
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

                // Vi använder en switch / case statement för att supportera extra paramterar för extract metoden att supportera både träning och test datasetet.

                case "extract":
                    new FeatureExtractor().Extract(args[1], args[2]);
                    break;
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