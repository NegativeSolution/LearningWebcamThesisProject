using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Text.Json;
using System.Net;

namespace ClipWebcamDetection
{
    class Program
    {
        static bool drawing = false;
        static Point start;
        static Rect box;

        static void Main(string[] args)
        {
            // 🔽 ADD THIS (ensures .data file exists next to model)
            ModelDownloader.EnsureDataFileForModel("Models/clip_image_encoder.onnx");

            // Load CLIP model + embeddings + labels from JSON
            ClipClient.Initialize("Models/clip_image_encoder.onnx", "text_embeddings.json");

            // Setup webcam
            VideoCapture cam = new VideoCapture(0, VideoCaptureAPIs.DSHOW);
            if (!cam.IsOpened())
            {
                Console.WriteLine("Camera failed.");
                return;
            }

            Window win = new Window("Camera");
            Cv2.SetMouseCallback("Camera", Mouse);

            Mat frame = new Mat();

            while (true)
            {
                cam.Read(frame);
                if (frame.Empty()) continue;

                Mat display = frame.Clone();

                if (box.Width > 0 && box.Height > 0)
                    Cv2.Rectangle(display, box, Scalar.Red, 2);

                win.ShowImage(display);

                int key = Cv2.WaitKey(20);

                if (key == 27) break; // ESC

                if (key == 32 && box.Width > 0) // SPACE
                {
                    Mat crop = new Mat(frame, box);

                    string predictedLabel = ClipClient.Predict(crop);

                    Console.WriteLine("Predicted: " + predictedLabel);
                    Console.Write("Enter a new label to add (or press Enter to skip): ");
                    string newLabel = Console.ReadLine()?.Trim();

                    if (!string.IsNullOrEmpty(newLabel))
                    {
                        ClipClient.AddNewLabel(newLabel, crop, "text_embeddings.json");
                    }

                    Metadata.Save(crop, predictedLabel);

                    box = new Rect();
                }
            }

            cam.Release();
            Cv2.DestroyAllWindows();
        }

        static void Mouse(MouseEventTypes e, int x, int y, MouseEventFlags f, IntPtr p)
        {
            if (e == MouseEventTypes.LButtonDown)
            {
                drawing = true;
                start = new Point(x, y);
            }

            if (e == MouseEventTypes.MouseMove && drawing)
            {
                box = new Rect(
                    Math.Min(start.X, x),
                    Math.Min(start.Y, y),
                    Math.Abs(x - start.X),
                    Math.Abs(y - start.Y));
            }

            if (e == MouseEventTypes.LButtonUp)
                drawing = false;
        }
    }

    // =========================
    // 🔽 DOWNLOADER ADDED
    // =========================
    static class ModelDownloader
    {
        public static void EnsureDataFileForModel(string modelPath)
        {
            string modelDir = Path.GetDirectoryName(modelPath)!;
            string dataPath = Path.Combine(modelDir, "clip_image_encoder.onnx.data");
            string tempPath = dataPath + ".tmp";

            string dataUrl = "https://github.com/NegativeSolution/LearningWebcamThesisProject/releases/download/v1.0/clip_image_encoder.onnx.data";

            if (File.Exists(dataPath))
            {
                long size = new FileInfo(dataPath).Length;
                if (size < 10_000_000)
                {
                    Console.WriteLine("Corrupted .data file detected. Redownloading...");
                    File.Delete(dataPath);
                }
            }

            if (!File.Exists(dataPath))
            {
                Console.WriteLine($"Downloading .onnx.data to: {modelDir}");

                try
                {
                    using (WebClient client = new WebClient())
                    {
                        client.DownloadProgressChanged += (s, e) =>
                        {
                            Console.Write($"\rProgress: {e.ProgressPercentage}%   ");
                        };

                        client.DownloadFile(dataUrl, tempPath);
                    }

                    File.Move(tempPath, dataPath, true);
                    Console.WriteLine("\nDownload complete.");
                }
                catch (Exception ex)
                {
                    if (File.Exists(tempPath))
                        File.Delete(tempPath);

                    Console.WriteLine("\nDownload failed:");
                    Console.WriteLine(ex.Message);
                    Environment.Exit(1);
                }
            }
            else
            {
                Console.WriteLine(".onnx.data already present.");
            }
        }
    }

    static class Metadata
    {
        static string root = "ImageBank";

        public static void Save(Mat img, string label)
        {
            string folder = Path.Combine(root, label);
            Directory.CreateDirectory(folder);

            string file = Path.Combine(folder, DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".png");
            Cv2.ImWrite(file, img);

            string meta = Path.Combine(root, "metadata.json");
            File.AppendAllText(meta, file + "|" + label + Environment.NewLine);
        }
    }

    class ClipEmbeddingFile
    {
        public List<string> labels { get; set; }
        public List<float[]> embeddings { get; set; }
        public List<List<float[]>> image_embeddings { get; set; }
    }

    static class ClipClient
    {
        static InferenceSession session;
        static List<string> labels = new List<string>();
        static List<float[]> textEmbeddings = new List<float[]>();
        static List<List<float[]>> imageEmbeddings = new List<List<float[]>>();
        static int imageSize = 224;

        public static void Initialize(string modelPath, string embeddingsPath)
        {
            session = new InferenceSession(modelPath);

            ClipEmbeddingFile data = null;
            if (File.Exists(embeddingsPath))
            {
                var json = File.ReadAllText(embeddingsPath);
                data = JsonSerializer.Deserialize<ClipEmbeddingFile>(json);
            }

            if (data != null)
            {
                labels = data.labels ?? new List<string>();
                textEmbeddings = data.embeddings ?? new List<float[]>();

                if (data.image_embeddings == null)
                    data.image_embeddings = new List<List<float[]>>();

                while (data.image_embeddings.Count < labels.Count)
                    data.image_embeddings.Add(new List<float[]>());

                imageEmbeddings = data.image_embeddings;
            }
            else
            {
                labels = new List<string>();
                textEmbeddings = new List<float[]>();
                imageEmbeddings = new List<List<float[]>>();
            }

            while (textEmbeddings.Count < labels.Count)
                textEmbeddings.Add(null);
            while (imageEmbeddings.Count < labels.Count)
                imageEmbeddings.Add(new List<float[]>());

            Console.WriteLine($"Loaded {labels.Count} labels and embeddings.");
        }

        public static string Predict(Mat image)
        {
            float[] imageEmbedding = EncodeImage(image);

            float bestScore = float.MinValue;
            int bestIndex = 0;

            for (int i = 0; i < labels.Count; i++)
            {
                float bestForLabel = float.MinValue;
                bool anyCompared = false;

                if (i < textEmbeddings.Count && textEmbeddings[i] != null && textEmbeddings[i].Length == imageEmbedding.Length)
                {
                    anyCompared = true;
                    float score = CosineSimilarity(imageEmbedding, textEmbeddings[i]);
                    if (score > bestForLabel) bestForLabel = score;
                }

                if (i < imageEmbeddings.Count && imageEmbeddings[i] != null)
                {
                    foreach (var ie in imageEmbeddings[i])
                    {
                        if (ie != null && ie.Length == imageEmbedding.Length)
                        {
                            anyCompared = true;
                            float score = CosineSimilarity(imageEmbedding, ie);
                            if (score > bestForLabel) bestForLabel = score;
                        }
                    }
                }

                if (!anyCompared) continue;

                if (bestForLabel > bestScore)
                {
                    bestScore = bestForLabel;
                    bestIndex = i;
                }
            }

            if (labels.Count == 0)
                return "unknown";

            Console.WriteLine($"Confidence: {bestScore:F3}");
            return labels[bestIndex];
        }

        public static void AddNewLabel(string label, Mat image, string embeddingsPath)
        {
            if (string.IsNullOrWhiteSpace(label)) return;

            int idx = labels.IndexOf(label);
            float[] embedding = EncodeImage(image);

            if (idx >= 0)
            {
                if (idx >= imageEmbeddings.Count) imageEmbeddings.Add(new List<float[]>());
                imageEmbeddings[idx].Add(embedding);
            }
            else
            {
                labels.Add(label);
                textEmbeddings.Add(null);
                var list = new List<float[]> { embedding };
                imageEmbeddings.Add(list);
            }

            var data = new ClipEmbeddingFile
            {
                labels = labels,
                embeddings = textEmbeddings,
                image_embeddings = imageEmbeddings
            };

            var options = new JsonSerializerOptions { WriteIndented = true };
            File.WriteAllText(embeddingsPath, JsonSerializer.Serialize(data, options));

            Console.WriteLine($"Saved embedding for label '{label}' to {embeddingsPath}");
        }

        static float CosineSimilarity(float[] a, float[] b)
        {
            if (a.Length != b.Length) throw new ArgumentException($"Embedding length mismatch: {a.Length} != {b.Length}");
            float dot = 0, magA = 0, magB = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                magA += a[i] * a[i];
                magB += b[i] * b[i];
            }
            return dot / (float)(Math.Sqrt(magA) * Math.Sqrt(magB));
        }

        static float[] EncodeImage(Mat image)
        {
            Mat resized = new Mat();
            Cv2.Resize(image, resized, new OpenCvSharp.Size(imageSize, imageSize));
            resized.ConvertTo(resized, MatType.CV_32FC3, 1.0 / 255);

            var tensor = new DenseTensor<float>(new[] { 1, 3, imageSize, imageSize });
            for (int y = 0; y < imageSize; y++)
            {
                for (int x = 0; x < imageSize; x++)
                {
                    Vec3f pixel = resized.At<Vec3f>(y, x);
                    tensor[0, 0, y, x] = pixel.Item2;
                    tensor[0, 1, y, x] = pixel.Item1;
                    tensor[0, 2, y, x] = pixel.Item0;
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", tensor)
            };

            using var results = session.Run(inputs);

            int expected = -1;
            var te = textEmbeddings.FirstOrDefault(e => e != null);
            if (te != null) expected = te.Length;
            else
            {
                foreach (var list in imageEmbeddings)
                {
                    var ie = list?.FirstOrDefault(e => e != null);
                    if (ie != null) { expected = ie.Length; break; }
                }
            }
            if (expected <= 0) expected = 512;

            foreach (var r in results)
            {
                var t = r.AsTensor<float>();
                if (t.Length == expected)
                    return t.ToArray();
            }

            string available = string.Join(", ", results.Select(r => $"{r.Name}:{r.AsTensor<float>().Length}"));
            throw new InvalidOperationException($"Could not find model output with length {expected}. Available outputs: {available}");
        }
    }
}