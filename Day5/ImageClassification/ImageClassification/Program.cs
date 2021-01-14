using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ImageClassification
{
    class Program
    {
        // Kataloger till träningsbilder, modelpath path:en dit den färdig tränade modellen existerar, samt savepath där vi sparar våran modell efter utfört transer learning.

        private static readonly string _hotDogTrainImagesPath = "..\\..\\..\\Data\\train\\hotdog";
        private static readonly string _pizzaTrainImagesPath = "..\\..\\..\\Data\\train\\pizza";
        private static readonly string _sushiTrainImagesPath = "..\\..\\..\\Data\\train\\sushi";
        private static readonly string _testImagesPath = "..\\..\\..\\Data\\test";
        private static readonly string _modelPath = "..\\..\\..\\Model\\tensorflow_inception_graph.pb";
        private static readonly string _savePath = "..\\..\\..\\Model\\hotdog.zip";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Ladda in träningsdata från tre olika kataloger med 10 bilder utav varje maträtt.
            var trainingData = new List<ImageData>();
            LoadImageData(trainingData, Path.GetFullPath(_hotDogTrainImagesPath), "hotdog");
            LoadImageData(trainingData, Path.GetFullPath(_pizzaTrainImagesPath), "pizza");
            LoadImageData(trainingData, Path.GetFullPath(_sushiTrainImagesPath), "sushi");

            /*
            1. Vi skapar en pipeline, laddar in dem tre olika katalogerna med träning för Hotdogs, Pizza och Sushi. 
            2. Sedan så ljusterar vi storleken på bilderna för att anpassa modellen med Inception inställningarna från våran class med samma namn.
            3. Färgerna görs sedan om till bilder i endast gråskala.
            4. Vi laddar in en redan färdig tränade TensorFlow modell som Google har tränat.
            5. På det så använder vi SoftMax funktionen för att aktivera inlärningen utav våra bilder som vi vill lära modellen att känna igen.
            6. Vi berättar sedan för modellen vad det är för något den ser på bilderna så den kan ge tillbaka ett namn.
            */
            var pipeline = context.Transforms.LoadImages(outputColumnName: "input", imageFolder: Path.GetFullPath(_hotDogTrainImagesPath), inputColumnName: "ImagePath")
                .Append(context.Transforms.LoadImages(outputColumnName: "input", imageFolder: Path.GetFullPath(_pizzaTrainImagesPath), inputColumnName: "ImagePath"))
                .Append(context.Transforms.LoadImages(outputColumnName: "input", imageFolder: Path.GetFullPath(_sushiTrainImagesPath), inputColumnName: "ImagePath"))
                .Append(context.Transforms.ResizeImages(outputColumnName: "input", inputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight))
                .Append(context.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                .Append(context.Model.LoadTensorFlowModel(_modelPath)
                    .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true)
                .Append(context.Transforms.Conversion.MapValueToKey(outputColumnName: "Key", inputColumnName: "Label"))
                .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Key", featureColumnName: "softmax2_pre_activation"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel")));

            // Vi tränar modellen genom att anropa fit metoden.
            Console.WriteLine("Training the model...");
            var data = context.Data.LoadFromEnumerable<ImageData>(trainingData); // Vi skapar en IDataView från en IEnumerable
            var model = pipeline.Fit(data);
            Console.WriteLine();

            // Vi testar modellen med tre bilder för dem tre olika maträtterna som vi tränar modellen att kunna lära sig. Modellen får berätta vad den tror det är samt sannolikheten.
            var files = Directory.EnumerateFiles(Path.GetFullPath(_testImagesPath));
            var predictor = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

            foreach (var file in files)
            {
                var image = new ImageData { ImagePath = file };
                var result = predictor.Predict(image);
                var label = result.PredictedLabel;
                var probability = result.Score.Max();
                Console.WriteLine($"{Path.GetFileName(file)} - {label} ({probability:P2})");
            }

            // Vi sparar ner modellen så att vi sedan kan använda den för att klassificera bilder. Som vi sedan kan återanvända i våran NotHotDog solution.
            Console.WriteLine();
            Console.WriteLine("Saving the model");
            context.Model.Save(model, data.Schema, _savePath);
        }

        // Vi laddar in bilder som vi sedan använder för att träna transfer learning modellen med.
        private static void LoadImageData(List<ImageData> images, string path, string label)
        {
            var files = Directory.EnumerateFiles(path);

            foreach (var file in files)
            {
                var imageData = new ImageData
                {
                    ImagePath = file,
                    Label = label
                };

                images.Add(imageData);
            }
        }
    }

    public class ImageData
    {
        public string ImagePath;
        public string Label;
    }

    public class ImagePrediction
    {
        public float[] Score;
        public string PredictedLabel;
    }

    // Vi sätter en aspect ratio eller storlek på bilderna som vi vill modifera i samband med träningen.
    public struct InceptionSettings
    {
        public const int ImageHeight = 224;
        public const int ImageWidth = 224;
        public const float Mean = 117;
        public const float Scale = 1;
        public const bool ChannelsLast = true;
    }
}