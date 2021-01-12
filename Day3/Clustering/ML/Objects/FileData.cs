using System;
using System.Linq;

using clustering.Enums;

using Microsoft.ML.Data;

namespace clustering.ML.Objects
{
    public class FileData
    {
        // Class:en FileData används för både att göra en prediction och träna våran data till modellen
        private const float TRUE = 1.0f;
        private const float FALSE = 0.0f;

        public FileData(Span<byte> data, string fileName = null)
        {
            // Använd för träningsändamål endast
            if (!string.IsNullOrEmpty(fileName))
            {
                if (fileName.Contains("ps1"))
                {
                    Label = (float) FileTypes.Script;
                } else if (fileName.Contains("exe"))
                {
                    Label = (float) FileTypes.Executable;
                } else if (fileName.Contains("doc"))
                {
                    Label = (float) FileTypes.Document;
                }
            }

            IsBinary = HasBinaryContent(data) ? TRUE : FALSE;

            IsMZHeader = HasHeaderBytes(data.Slice(0, 2), "MZ") ? TRUE : FALSE;

            IsPKHeader = HasHeaderBytes(data.Slice(0, 2), "PK") ? TRUE : FALSE;
        }

        /// <summary>
        /// Använd för att mappa cluster id's till resultat
        /// </summary>
        /// <param name="fileType"></param>
        public FileData(FileTypes fileType)
        {
            Label = (float)fileType;

            switch (fileType)
            {
                case FileTypes.Document:
                    IsBinary = TRUE;
                    IsMZHeader = FALSE;
                    IsPKHeader = TRUE;
                    break;
                case FileTypes.Executable:
                    IsBinary = TRUE;
                    IsMZHeader = TRUE;
                    IsPKHeader = FALSE;
                    break;
                case FileTypes.Script:
                    IsBinary = FALSE;
                    IsMZHeader = FALSE;
                    IsPKHeader = FALSE;
                    break;
            }
        }

        private static bool HasBinaryContent(Span<byte> fileContent) =>
            System.Text.Encoding.UTF8.GetString(fileContent.ToArray()).Any(a => char.IsControl(a) && a != '\r' && a != '\n');

        private static bool HasHeaderBytes(Span<byte> data, string match) => System.Text.Encoding.UTF8.GetString(data) == match;

        // Vi använder properties attributen som används för prediction, träning och testning:

        [ColumnName("Label")]
        public float Label { get; set; }

        public float IsBinary { get; set; }

        public float IsMZHeader { get; set; }

        public float IsPKHeader { get; set; }

        // Vi använde ToString metoden för att användas med feature extraction.
        public override string ToString() => $"{Label},{IsBinary},{IsMZHeader},{IsPKHeader}";
    }
}