
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Inference;
using Grpc.Core;
using System.Drawing;
using System.IO;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using Google.Protobuf;


namespace TritonNET
{

    internal class Program
    {
        
        static void Main(string[] args)
        {

            string modelName = "efficientdet_d0";
            string modelVersion = "1";
            string server_url = "172.16.10.15";
            string server_port = "8001";
            float confidence_threshold = 0.4f;

            ModelMetadataResponse modelMetadata;

            using (TritonNETClient tritonClient = new TritonNETClient(server_url, server_port))
            {
                tritonClient.Connect();

                var labels = GetLabels();

                modelMetadata = tritonClient.ModelInfo(modelName, modelVersion).Result;

                var image = Image.FromFile("examples/elephant.jpg");

                var inferResultTask = tritonClient.ModelInfer(modelMetadata, GetData(image), 512, 512);
                //var modelInfo = tritonClient.ModelInfo("efficientdet_d0", "1");

                var detections = PostProcess(inferResultTask.Result, labels);

                PrintResult(detections, image);

                Console.ReadKey();

            }

            void PrintResult(ICollection<DetectionObject> detections, Image image)
            {

                if(detections.Count > 0)
                {
                    using (Pen pen = new Pen(Color.DarkCyan, 2))
                    using (Graphics graphics = Graphics.FromImage(image))
                    {

                        foreach (DetectionObject detection in detections)
                        {

                            var y_min = image.Height * detection.y_min;
                            var x_min = image.Width * detection.x_min;
                            var y_max = image.Height * detection.y_max;
                            var x_max = image.Width * detection.x_max;

                            var box_height = y_max - y_min;
                            var box_width = x_max - x_min;

                            var box_center_x = x_min + (box_width / 2);
                            var box_center_y = y_min + (box_height / 2);

                            Rectangle box = new Rectangle((int)x_min, (int)y_min, (int)box_width, (int)box_height);

                            Console.WriteLine($"Drawing detected label: {detection.label}");

                            graphics.DrawRectangle(pen, box);


                        }

                        image.Save("examples/out.jpg");
                    }
                }
                else
                {
                    Console.WriteLine("No detections.");
                }

                
            }

            List<DetectionObject> PostProcess(ModelInferResponse inferResponse, List<string> labels)
            {

                var outputs = inferResponse.Outputs;


                //Find the index of "detection_scores" in outputs - to be able to filter by threshold
                var scores = outputs.Where(o => o.Name == "detection_scores").First();
                var scores_index = outputs.IndexOf(scores);

                var det_scores_bytes = inferResponse.RawOutputContents[scores_index].ToByteArray();
                var det_scores = new float[det_scores_bytes.Length / 4];
                Buffer.BlockCopy(det_scores_bytes, 0, det_scores, 0, det_scores_bytes.Length);

                //The number of filtered scores
                var num_filtered_classes = 0;
                for (int i = 0; i < det_scores.Length; i++)
                {
                    if (det_scores[i] >= confidence_threshold)
                    {
                        num_filtered_classes++;
                    }
                    else
                        break;
                }

                var detectedObjects = new List<DetectionObject>();

                //Any detections?
                if (num_filtered_classes > 0)
                {

                    //Find the index of "detection_classes" in outputs
                    var classes = outputs.Where(o => o.Name == "detection_classes").First();
                    var classes_index = outputs.IndexOf(classes);

                    var det_classes_bytes = inferResponse.RawOutputContents[classes_index].ToByteArray();
                    var det_classes = new float[num_filtered_classes];

                    Buffer.BlockCopy(det_classes_bytes, 0, det_classes, 0, num_filtered_classes * 4);

                    //Find the index of "detection_boxes" in outputs
                    var boxes = outputs.Where(o => o.Name == "detection_boxes").First();
                    var boxes_index = outputs.IndexOf(boxes);

                    var det_boxes_bytes = inferResponse.RawOutputContents[boxes_index].ToByteArray();
                    var det_boxes = new float[num_filtered_classes * 4];

                    Buffer.BlockCopy(det_boxes_bytes, 0, det_boxes, 0, num_filtered_classes * 4 * 4);

                    for (int i = 0; i < num_filtered_classes; i++)
                    {

                        var box_coordinates = det_boxes.Skip(i * 4).Take(4).ToArray();

                        var ob = new DetectionObject(labels[(int)det_classes[i] - 1], det_scores[i], box_coordinates);


                        detectedObjects.Add(ob);

                    }

                }
                
                return detectedObjects; 



                //Find the index of "detection_classes" in outputs
                /*                var classes = outputs.Where(o => o.Name == "detection_classes").First();
                                var classes_index = outputs.IndexOf(classes);

                                var det_classes_bytes = inferResponse.RawOutputContents[classes_index].ToByteArray();
                                var det_classes = new float[det_classes_bytes.Length / 4];
                                Buffer.BlockCopy(det_classes_bytes, 0, det_classes, 0, det_classes_bytes.Length);*/



                /*                var det_scores_bytes = output_classes.RawOutputContents[3].ToByteArray();
                                var det_scores = new float[det_scores_bytes.Length / 4];
                                Buffer.BlockCopy(det_scores_bytes, 0, det_scores, 0, det_scores_bytes.Length);

                                var det_multiclass_bytes = output_classes.RawOutputContents[2].ToByteArray();
                                var det_multiclass_scores = new float[det_multiclass_bytes.Length / 4];
                                Buffer.BlockCopy(det_multiclass_bytes, 0, det_multiclass_scores, 0, det_multiclass_bytes.Length);*/

                //Console.WriteLine($"Detected: {labels[((int)det_classes[0]) - 1]}, {labels[((int)det_classes[1]) - 1]}");

            }

            byte[] GetData(Image image)
            {
                
                
                var image2 = ResizeImage(image,512,512);

                using (MemoryStream ms = new MemoryStream())
                {

                    image2.Save(ms, ImageFormat.Jpeg);
                    //return ms.ToArray();
                }

                byte[] pixel_data = new byte[image2.Width*image2.Height*3];
                int pix_set = 0;
                for (int i = 0; i < image2.Width; i++)
                {
                    for (int j = 0; j < image2.Height; j++)
                    {

                        var pixel = image2.GetPixel(j, i);

                        var r = pixel.R;
                        var g = pixel.G;
                        var b = pixel.B;

                        pixel_data[pix_set] = r;
                        pixel_data[pix_set+1] = g;
                        pixel_data[pix_set+2] = b;

                        pix_set += 3;
                    }
                }

                return pixel_data;
            }

            List<string> GetLabels()
            {


                //var path = "examples/labels.txt";
                var path = "examples/mscoco_labels_original.txt";

                var labels = File.ReadLines(path, Encoding.UTF8);


                return labels.ToList();
            }

            List<string> GetLabels2()
            {


                var path = "examples/labels.txt";

                var labels = File.ReadLines(path, Encoding.UTF8);


                return labels.ToList();
            }

            Bitmap ResizeImage(Image image, int width, int height)
            {
                var destRect = new Rectangle(0, 0, width, height);
                var destImage = new Bitmap(width, height);

                destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

                using (var graphics = Graphics.FromImage(destImage))
                {
                    graphics.CompositingMode = CompositingMode.SourceCopy;
                    graphics.CompositingQuality = CompositingQuality.HighQuality;
                    graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    graphics.SmoothingMode = SmoothingMode.HighQuality;
                    graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                    using (var wrapMode = new ImageAttributes())
                    {
                        wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                        graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                    }
                }

                return destImage;
            }


        }

    }
}
