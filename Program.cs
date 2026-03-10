using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Text.Json;

namespace ClipWebcamDetection
{
    class Program
    {
        static bool drawing = false;
        static Point start;
        static Rect box;

        static void Main(string[] args)
        {
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

                    string label = ClipClient.Predict(crop);

                    Metadata.Save(crop, label);

                    Console.WriteLine("Saved: " + label);

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

    static class Metadata
    {
        static string root = "ImageBank";

        public static void Save(Mat img, string label)
        {
            string folder = Path.Combine(root, label);

            Directory.CreateDirectory(folder);

            string file =
                Path.Combine(folder,
                DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".png");

            Cv2.ImWrite(file, img);

            string meta = Path.Combine(root, "metadata.json");

            File.AppendAllText(meta, file + "|" + label + Environment.NewLine);
        }
    }

    // Structure of the JSON file
    class ClipEmbeddingFile
    {
        public List<string> labels { get; set; }
        public List<float[]> embeddings { get; set; }
    }

    static class ClipClient
    {
        static InferenceSession session;

        static List<string> labels = new List<string>();

        static List<float[]> textEmbeddings = new List<float[]>();

        static int imageSize = 224;

        public static void Initialize(string modelPath, string embeddingsPath)
        {
            // Load ONNX model
            session = new InferenceSession(modelPath);

            // Load JSON
            var json = File.ReadAllText(embeddingsPath);

            var data = JsonSerializer.Deserialize<ClipEmbeddingFile>(json);

            labels = data.labels;

            textEmbeddings = data.embeddings;

            Console.WriteLine($"Loaded {labels.Count} labels and embeddings.");
        }

        public static string Predict(Mat image)
        {
            float[] imageEmbedding = EncodeImage(image);

            // Validate embedding dimension
            if (textEmbeddings.Count > 0 && imageEmbedding.Length != textEmbeddings[0].Length)
                throw new InvalidOperationException($"Embedding size mismatch: image {imageEmbedding.Length} vs text {textEmbeddings[0].Length}");

            float bestScore = float.MinValue;
            int bestIndex = 0;

            for (int i = 0; i < textEmbeddings.Count; i++)
            {
                float score = CosineSimilarity(imageEmbedding, textEmbeddings[i]);

                if (score > bestScore)
                {
                    bestScore = score;
                    bestIndex = i;
                }
            }

            Console.WriteLine($"Confidence: {bestScore:F3}");

            return labels[bestIndex];
        }

        // Safer CosineSimilarity (throws on length mismatch)
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
            // Resize & normalize
            Mat resized = new Mat();
            Cv2.Resize(image, resized, new OpenCvSharp.Size(imageSize, imageSize));
            resized.ConvertTo(resized, MatType.CV_32FC3, 1.0 / 255);

            // Prepare tensor [1,3,H,W]
            var tensor = new DenseTensor<float>(new[] { 1, 3, imageSize, imageSize });
            for (int y = 0; y < imageSize; y++)
            {
                for (int x = 0; x < imageSize; x++)
                {
                    Vec3f pixel = resized.At<Vec3f>(y, x);
                    tensor[0, 0, y, x] = pixel.Item2; // R
                    tensor[0, 1, y, x] = pixel.Item1; // G
                    tensor[0, 2, y, x] = pixel.Item0; // B
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", tensor)
            };

            using var results = session.Run(inputs);

            int expected = textEmbeddings.Count > 0 ? textEmbeddings[0].Length : 512;

            // Pick output whose length matches text embedding
            foreach (var r in results)
            {
                var t = r.AsTensor<float>();
                if (t.Length == expected)
                    return t.ToArray();
            }

            // If none found, throw clear exception with available outputs
            string available = string.Join(", ", results.Select(r => $"{r.Name}:{r.AsTensor<float>().Length}"));
            throw new InvalidOperationException($"Could not find model output with length {expected}. Available outputs: {available}");
        }
    }
}