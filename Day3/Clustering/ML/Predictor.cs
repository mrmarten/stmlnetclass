using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using clustering.Enums;
using clustering.ML.Base;
using clustering.ML.Objects;

using Microsoft.ML;

namespace clustering.ML
{
    public class Predictor : BaseML
    {
        /*
        Först så lägger vi till en helper metod, GetClusterToMap som mappar kända värden till
        predektionerna för klustren. Vi använder Enum.GetValues, allt eftersom du lägger till fler types, så behövs metoden inte bli modiferad.

        */
        private Dictionary<uint, FileTypes> GetClusterToMap(PredictionEngineBase<FileData, FileTypePrediction> predictionEngine)
        {
            var map = new Dictionary<uint, FileTypes>();

            var fileTypes = Enum.GetValues(typeof(FileTypes)).Cast<FileTypes>();

            foreach (var fileType in fileTypes)
            {

                var fileData = new FileData(fileType);

                var prediction = predictionEngine.Predict(fileData);

                map.Add(prediction.PredictedClusterId, fileType);
            }

            return map;
        }

        public void Predict(string inputDataFile)
        {
            if (!File.Exists(ModelPath))
            {
                Console.WriteLine($"Failed to find model at {ModelPath}");

                return;
            }

            if (!File.Exists(inputDataFile))
            {
                Console.WriteLine($"Failed to find input data at {inputDataFile}");

                return;
            }

            ITransformer mlModel;
            
            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = MlContext.Model.Load(stream, out _);
            }

            if (mlModel == null)
            {
                Console.WriteLine("Failed to load model");

                return;
            }

            /*
            Vi passar sen in FileData och FileTypePrediction typerna in till
            CreatePredictionEngine metoden och skapar våran production engine.
            Sedan så läser vi in den binära filen och passar in data till constructorn utav FileData
            innan körning utav prediction och att mappningen har blivit etablerade.
             */

            var predictionEngine = MlContext.Model.CreatePredictionEngine<FileData, FileTypePrediction>(mlModel);

            var fileData = new FileData(File.ReadAllBytes(inputDataFile));

            var prediction = predictionEngine.Predict(fileData);

            var mapping = GetClusterToMap(predictionEngine);

            // Sedan så ljusterar vi outputen så att det matchar outputen utav k-means klustrerts beräknings resultat, som mäts i distansen Euclidean distance.

            Console.WriteLine(
                $"Based on input file: {inputDataFile}{Environment.NewLine}{Environment.NewLine}" +
                $"Feature Extraction: {fileData}{Environment.NewLine}{Environment.NewLine}" +
                $"The file is predicted to be a {mapping[prediction.PredictedClusterId]}{Environment.NewLine}");

            Console.WriteLine("Distances from all clusters:");

            for (uint x = 0; x < prediction.Distances.Length; x++) { 
                Console.WriteLine($"{mapping[x+1]}: {prediction.Distances[x]}");
            }
        }
    }
}