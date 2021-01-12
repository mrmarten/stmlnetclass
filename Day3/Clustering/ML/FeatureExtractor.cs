using System;
using System.IO;

using clustering.Common;
using clustering.ML.Base;
using clustering.ML.Objects;

namespace clustering.ML
{
    public class FeatureExtractor : BaseML
    {
        /* 
         FeatureExtractor använder sig utav logistic regression för både träning och test data.
         Först så standardsätter vi generalisering utav extrahering utav mappsökvägen och filen vi skriver till
         Vi passar också filnamnet som ger Labeling att inträffa innuti FileData class:en 
        */
        private void ExtractFolder(string folderPath, string outputFile)
        {
            if (!Directory.Exists(folderPath))
            {
                Console.WriteLine($"{folderPath} does not exist");

                return;
            }

            var files = Directory.GetFiles(folderPath);

            using (var streamWriter =
                new StreamWriter(Path.Combine(AppContext.BaseDirectory, $"../../../Data/{outputFile}")))
            {
                foreach (var file in files)
                {
                    var extractedData = new FileData(File.ReadAllBytes(file), file);

                    streamWriter.WriteLine(extractedData.ToString());
                }
            }

            Console.WriteLine($"Extracted {files.Length} to {outputFile}");
        }

        // Sist så använder vi två paramterar från command linen som är anropas från Program class:En och sedan andropar metoden för en andra gång.

        public void Extract(string trainingPath, string testPath)
        {
            ExtractFolder(trainingPath, Constants.SAMPLE_DATA);
            ExtractFolder(testPath, Constants.TEST_DATA);
        }
    }
}