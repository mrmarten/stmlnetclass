using Microsoft.ML.Data;

namespace recommendation.ML.Objects
{

    /*
    MusicRating class:en är en kontainer clas som innehåller datat för både
    beräkna prediction men också att träna våran modell. Numret i LoadColumn mappar till index
    värdet i CSV filen. Matrix Faktorisering i ML.NET kräver användandet utav nomalisering som
    visas i nedstående kod-block.
    */
    public class MusicRating
    {
        [LoadColumn(0)]
        public float UserID { get; set; }

        [LoadColumn(1)]
        public float MusicID { get; set; }

        [LoadColumn(2)]
        public float Label { get; set; }
    }
}