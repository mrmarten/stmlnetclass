using Microsoft.ML;
using Microsoft.Win32;
using System;
using System.Linq;
using System.Windows;
using System.Windows.Media.Imaging;

namespace NotHotDog
{
    /// <summary>
    
    /// </summary>
    public partial class MainWindow : Window
    {
        private PredictionEngine<ImageData, ImagePrediction> _predictor;
        // Vi sätter path:en till våran tränade modell från Transfer Learning träningen från våran tidigare ImageClassfication solution.
        private static readonly string _modelPath = "..\\..\\..\\Model\\hotdog.zip";

        public MainWindow()
        {
            InitializeComponent();
            this.Loaded += OnLoaded;
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            // Vi laddar in våran tränade ML.NET modell i zip format.
            var context = new MLContext(seed: 0);
            var model = context.Model.Load(_modelPath, out DataViewSchema schema);
            _predictor = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
        }

        private void OnSelectImageButtonClicked(object sender, RoutedEventArgs e)
        {
            // Vi skapar en dialog så vi kan välja och ladda in en fil
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "JPEG Files (*.jpg;*.jpeg)|*.jpg;*.jpeg|PNG Files (*.png)|*.png|All Files (*.*)|*.*";
            dialog.FilterIndex = 3;

            if (dialog.ShowDialog() == true)
            {
                string path = dialog.FileName;

                // Show the image
                LoadedImage.Source = new BitmapImage(new Uri(path));

                try
                {
                    // Vi använder sedan ML.NET för att bekräfta vad bildfilen innehåller.
                    var image = new ImageData { ImagePath = path };
                    var result = _predictor.Predict(image);
                    var label = result.PredictedLabel;
                    var probability = result.Score.Max();

                    if (String.Compare(label, "hotdog", true) == 0)
                    {
                        MessageBox.Show($"It's a hot dog! ({(probability * 100):0.#}%)");
                    }
                    else
                    {
                        MessageBox.Show($"Not a hot dog. Looks more like {label}.");
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
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
}