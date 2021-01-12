using System;
using System.Collections.Generic;
using System.IO;
using recommendation.Common;
using recommendation.ML.Base;
using recommendation.ML.Objects;

using Microsoft.ML;

using Newtonsoft.Json;

namespace recommendation.ML
{
    public class Predictor : BaseML
    {
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

            // Vi skapar våran prediktions motor med MusicRAting och MusicPrediction typer följande

            var predictionEngine = MlContext.Model.CreatePredictionEngine<MusicRating, MusicPrediction>(mlModel);

            // V läser inputen till ett string object.
            var json = File.ReadAllText(inputDataFile);

            // Vi dezaliserar sedan strängen intill ett objekt utav typen MusicRating
            var rating = JsonConvert.DeserializeObject<MusicRating>(json);

            // Sedan så startar vi själva prediktionen och alla output resultat från modellen körs.
            var prediction = predictionEngine.Predict(rating);

            Console.WriteLine(
                $"Based on input:{System.Environment.NewLine}" +
                $"Label: {rating.Label} | MusicID: {rating.MusicID} | UserID: {rating.UserID}{System.Environment.NewLine}" +
                $"The music is {(prediction.Score > Constants.SCORE_THRESHOLD ? "recommended" : "not recommended")}");
        }
    }
}