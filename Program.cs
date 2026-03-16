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
        // Text embeddings aligned by index with labels (may be null if not available)
        public List<float[]> embeddings { get; set; }
        // Per-label list of saved image embeddings (can be empty lists)
        public List<List<float[]>> image_embeddings { get; set; }
    }

    static class ClipClient
    {
        static InferenceSession session;
        static List<string> labels = new List<string>();
        // text embeddings aligned by index with labels. Entries may be null if not present.
        static List<float[]> textEmbeddings = new List<float[]>();
        // per-label image embeddings
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

                // Ensure image_embeddings exists and aligns with labels
                if (data.image_embeddings == null)
                    data.image_embeddings = new List<List<float[]>>();

                // Pad image_embeddings to match labels count
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

            // Ensure lists align in length
            while (textEmbeddings.Count < labels.Count)
                textEmbeddings.Add(null);
            while (imageEmbeddings.Count < labels.Count)
                imageEmbeddings.Add(new List<float[]>());

            Console.WriteLine($"Loaded {labels.Count} labels and embeddings.");
        }

        public static string Predict(Mat image)
        {
            float[] imageEmbedding = EncodeImage(image);

            // Iterate over labels (not textEmbeddings count) and compute best score per label
            float bestScore = float.MinValue;
            int bestIndex = 0;

            for (int i = 0; i < labels.Count; i++)
            {
                float bestForLabel = float.MinValue;
                bool anyCompared = false;

                // Compare against text embedding if present and same length
                if (i < textEmbeddings.Count && textEmbeddings[i] != null && textEmbeddings[i].Length == imageEmbedding.Length)
                {
                    anyCompared = true;
                    float score = CosineSimilarity(imageEmbedding, textEmbeddings[i]);
                    if (score > bestForLabel) bestForLabel = score;
                }

                // Compare against any saved image embeddings for this label
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

                // If no comparable embeddings for this label (length mismatch), skip it
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

            // Ensure lists align
            int idx = labels.IndexOf(label);
            float[] embedding = EncodeImage(image);

            if (idx >= 0)
            {
                // add image embedding to existing label
                if (idx >= imageEmbeddings.Count) imageEmbeddings.Add(new List<float[]>());
                imageEmbeddings[idx].Add(embedding);
            }
            else
            {
                // new label: append label, ensure arrays stay aligned
                labels.Add(label);

                // keep text embedding list aligned (no text embedding provided)
                textEmbeddings.Add(null);

                // add image embedding list for this new label
                var list = new List<float[]> { embedding };
                imageEmbeddings.Add(list);
            }

            // save back to JSON
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

            // Determine expected length from existing embeddings if possible:
            int expected = -1;
            // prefer a non-null text embedding length
            var te = textEmbeddings.FirstOrDefault(e => e != null);
            if (te != null) expected = te.Length;
            else
            {
                // fallback to any existing saved image embedding
                foreach (var list in imageEmbeddings)
                {
                    var ie = list?.FirstOrDefault(e => e != null);
                    if (ie != null) { expected = ie.Length; break; }
                }
            }
            // if still unknown, default to 512 (legacy behavior)
            if (expected <= 0) expected = 512;

            // Pick output whose length matches expected embedding length
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