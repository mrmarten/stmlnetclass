using System;
using System.IO;

using clustering.Common;

using Microsoft.ML;

namespace clustering.ML.Base
{
    public class BaseML
    {

        //Här skapar vi våran Featues variabel som vi sedan använder för tränings Class:en
        //Även den fulla sökvägen till var modellen finns sparad.
        protected const string FEATURES = "Features";

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, Constants.MODEL_FILENAME);

        protected readonly MLContext MlContext;

        protected BaseML()
        {
            MlContext = new MLContext(2020);
        }
    }
}