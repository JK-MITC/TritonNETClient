using Inference;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TritonNET
{
    internal class DetectionObject
    {
        public string label;
        public float confidence;

        //Y_MIN, X_MIN, Y_MAX, X_MAX
        float[] box_coordinates = new float[4];
        public float y_min { get; set; }
        public float x_min { get; set; }
        public float y_max { get; set; }
        public float x_max { get; set; }
        

        public DetectionObject(string label,float confidence,float[] box_coordinates)
        {

            this.label = label;
            this.confidence = confidence;
            this.box_coordinates = box_coordinates;
            this.y_min = box_coordinates[0];
            this.x_min = box_coordinates[1];
            this.y_max = box_coordinates[2];
            this.x_max = box_coordinates[3];
        }


    }
}
