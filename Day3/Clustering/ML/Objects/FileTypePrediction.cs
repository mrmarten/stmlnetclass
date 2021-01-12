using Microsoft.ML.Data;

namespace clustering.ML.Objects
{
    /*  FileTypePrediction class:en innehåller värden mappade till våran prediction output
        För k-means clustering så är PredictedClusterId värdet som innehålller det närmalste klustret funnet.
        som tillägg är det Distances array som innehåller distanserna från dem olika data punkterna till clustret
    */
    public class FileTypePrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}