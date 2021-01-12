using System;

using clustering.ML.Base;
using clustering.ML.Objects;

using Microsoft.ML;
using Microsoft.ML.Data;

namespace clustering.ML
{
    public class Trainer : BaseML
    {
        private IDataView GetDataView(string fileName)
        {
            // Vi använder GetDataView helper metoden som bygger på IDataView objektet från columnerna som vi tidigare definerade i FileData class:en.

            return MlContext.Data.LoadFromTextFile(path: fileName,
                columns: new[]
                {
                    new TextLoader.Column(nameof(FileData.Label), DataKind.Single, 0),
                    new TextLoader.Column(nameof(FileData.IsBinary), DataKind.Single, 1),
                    new TextLoader.Column(nameof(FileData.IsMZHeader), DataKind.Single, 2),
                    new TextLoader.Column(nameof(FileData.IsPKHeader), DataKind.Single, 3)
                },
                hasHeader: false,
                separatorChar: ',');
        }

        public void Train(string trainingFileName, string testingFileName)
        {
            if (!System.IO.File.Exists(trainingFileName))
            {
                Console.WriteLine($"Failed to find training data file ({trainingFileName}");

                return;
            }

            if (!System.IO.File.Exists(testingFileName))
            {
                Console.WriteLine($"Failed to find test data file ({testingFileName}");

                return;
            }

            // Vi bygger sedan en data processerande pipeline som transformerar columnen till enskilda Feature kolumner.
            var trainingDataView = GetDataView(trainingFileName);

            var dataProcessPipeline = MlContext.Transforms.Concatenate(
                FEATURES,
                nameof(FileData.IsBinary),
                nameof(FileData.IsMZHeader),
                nameof(FileData.IsPKHeader));
            
            // Vi fortsätter sedan att skapa en k-means tränare utav kluster storlek 3 och skapar en modell.

            var trainer = MlContext.Clustering.Trainers.KMeans(featureColumnName: FEATURES, numberOfClusters: 3);
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            MlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);


            // Vi utvärderar nu evaluering utav modellen som vi nyss tränat med hjälp utav test datasetet.

            var testingDataView = GetDataView(testingFileName);

            IDataView testDataView = trainedModel.Transform(testingDataView);

            // Till slut så printar vi ut all output från mätvärdena från klassificerings beräknaren.

            ClusteringMetrics modelMetrics = MlContext.Clustering.Evaluate(
                data: testDataView,
                labelColumnName: "Label",
                scoreColumnName: "Score",
                featureColumnName: FEATURES);

            Console.WriteLine($"Average Distance: {modelMetrics.AverageDistance}");
            Console.WriteLine($"Davies Bould Index: {modelMetrics.DaviesBouldinIndex}");
            Console.WriteLine($"Normalized Mutual Information: {modelMetrics.NormalizedMutualInformation}");
        }
    }
}