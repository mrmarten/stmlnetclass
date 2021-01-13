using System;
using System.IO;

using recommendation.ML.Base;
using recommendation.ML.Objects;

using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace recommendation.ML
{
    public class Trainer : BaseML
    {

        /* Vi lägger först till två konstanta variablar med variabel enkodning.
         Skapa en fil i bin katalogen input.json och lägg följande rad, sen spara filen
         
        { "UserID": 10, "MusicID": 3, "Label": 2 }
      
         Testa filen med kommandoprompten genom anropa exe filen och passa argument från json filen : Recommendation.exe predict input.json
          
         * */
        private const string UserIDEncoding = "UserIDEncoding";

        private const string MusicIDEncoding = "MusicIDEncoding";

        private (IDataView DataView, IEstimator<ITransformer> Transformer) GetDataView(string fileName, bool training = true)
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<MusicRating>(fileName, ',');

            if (!training)
            {
                return (trainingDataView, null);
            }

            IEstimator<ITransformer> dataProcessPipeline =
                MlContext.Transforms.Conversion.MapValueToKey(outputColumnName: UserIDEncoding,
                        inputColumnName: nameof(MusicRating.UserID))
                    .Append(MlContext.Transforms.Conversion.MapValueToKey(outputColumnName: MusicIDEncoding,
                        inputColumnName: nameof(MusicRating.MusicID)));

            return (trainingDataView, dataProcessPipeline);
        }

        public void Train(string trainingFileName, string testingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Failed to find training data file ({trainingFileName}");

                return;
            }

            if (!File.Exists(testingFileName))
            {
                Console.WriteLine($"Failed to find test data file ({testingFileName}");

                return;
            }

            var trainingDataView = GetDataView(trainingFileName);

            /* 
            Vi bygger sedan options för MatrixFactorizationTRainer. 
            För Row och Column namnen som vi tidigare definerat
            Quiet flaggan visar ytterliggare model byggangde information
            för varje iteration.
            */
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = UserIDEncoding,
                MatrixRowIndexColumnName = MusicIDEncoding,
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 10,
                Quiet = false
            };

            // Vi skapar en matrix faktoriserings tränare

            var trainingPipeLine = trainingDataView.Transformer.Append(MlContext.Recommendation().Trainers.MatrixFactorization(options));

            // Sedan startar vi träningen genom att köra Fit metoden, vi sparar sedan ner vår tränade modell.

            ITransformer trainedModel = trainingPipeLine.Fit(trainingDataView.DataView);

            MlContext.Model.Save(trainedModel, trainingDataView.DataView.Schema, ModelPath);

            Console.WriteLine($"Model saved to {ModelPath}{Environment.NewLine}");

            // Vi laddar och testar datat och passar data till matrix faktoriserings evulatorn.

            var testingDataView = GetDataView(testingFileName, true);

            var testSetTransform = trainedModel.Transform(testingDataView.DataView);

            var modelMetrics = MlContext.Recommendation().Evaluate(testSetTransform);

            /*
            Mean Squared Error vill försöka uppnå ett sånt långt värde som möjligt.
            Root Mean Squared Errror vill vi ha ett värde under 180 för att det ska räknas som en effektiv modell.
            */

            Console.WriteLine($"Matrix Factorization Evaluation:{Environment.NewLine}{Environment.NewLine}" +
                              $"Loss Function: {modelMetrics.LossFunction:F3}{Environment.NewLine}" +
                              $"Mean Absolute Error: {modelMetrics.MeanAbsoluteError:F3}{Environment.NewLine}" +
                              $"Mean Squared Error: {modelMetrics.MeanSquaredError:F3}{Environment.NewLine}" +
                              $"R Squared: {modelMetrics.RSquared:F3}{Environment.NewLine}" +
                              $"Root Mean Squared Error: {modelMetrics.RootMeanSquaredError:F3}");
        }
    }
}