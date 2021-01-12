namespace recommendation.ML.Objects
{
    /*
     I MusicPrediction class:en innehåller värdena som är mappade till prediction outputen.
     Poäng värdet är sannolikheten att prediktionen är korrekt. 
     */
    public class MusicPrediction
    {
        public float Label { get; set; }

        public float Score { get; set; }
    }
}