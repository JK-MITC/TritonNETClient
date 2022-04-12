using Google.Protobuf;
using Google.Protobuf.Collections;
using Grpc.Core;
using Inference;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Inference.ModelInferRequest.Types;

namespace TritonNET
{
    internal class TritonNETClient : IDisposable
    {
        private string url;
        private string port;
        private Channel channel;
        private GRPCInferenceService.GRPCInferenceServiceClient client;
 
        public TritonNETClient(string url,string port)
        {
            this.url = url;
            this.port = port;
            
        }

        public void Connect()
        {
            try
            {

                var channelOptions = new ChannelOption[] { new ChannelOption(ChannelOptions.MaxReceiveMessageLength, 99999999) };

                this.channel = new Channel(this.url +':'+ this.port, ChannelCredentials.Insecure,channelOptions);
                
                this.client = new GRPCInferenceService.GRPCInferenceServiceClient(channel);
                
                
            }
            catch (Exception ex)
            {

            }

        }

        public async Task<ModelMetadataResponse> ModelInfo(string model, string version)
        {
            if (channel != null && (channel.State == ChannelState.Ready || channel.State == ChannelState.Idle)) {

                try
                {

                    var modelInfoReq = new ModelMetadataRequest()
                    {
                        Name = model,
                        Version = version
                    };

                    var modelInfoResp = await client.ModelMetadataAsync(modelInfoReq);

                    return modelInfoResp;

                }
                catch (Exception ex)
                {

                    throw ex;
                }
            
            }
            else
            {
                throw new Exception("Channel not ready");
            }

        }

        public async Task<ModelInferResponse> ModelInfer(ModelMetadataResponse modelInfo, byte[] inputData, int width, int height)
        {
            if (channel != null && (channel.State == ChannelState.Ready || channel.State == ChannelState.Idle))
            {
                try{

                    //var dt = ByteString.CopyFrom(inputData);

                    //var uints = new uint[inputData.Length];

                    //for (int i = 0; i < uints.Length; i++)
                    //{
                    //    uints[i] = inputData[i];
                    //}


                    //var data = BitConverter.ToUInt16(inputData, 0);
                    //var conts = new InferTensorContents()
                    //{
                    //    UintContents = {uints}
                    //};



                    var input = new InferInputTensor() {
                        Name = modelInfo.Inputs[0].Name,
                        Datatype = modelInfo.Inputs[0].Datatype,//TritonTypeMap.TypeMap[modelInfo.Inputs[0].Datatype],
                        //Contents = conts,
                        Shape = { 1, width, height, 3 }//{1,512,512,3}
                    };

                    var outputs = new List<InferRequestedOutputTensor>();

                    foreach (var item in modelInfo.Outputs)
                    {
                        outputs.Add(new InferRequestedOutputTensor()
                        {

                            Name = item.Name,
                        });
                    }

                    /*                    var num_detections = new InferRequestedOutputTensor() { Name = "num_detections" };
                                        var det_boxes = new InferRequestedOutputTensor() { Name = "detection_boxes" };
                                        var det_scores = new InferRequestedOutputTensor() { Name = "detection_scores" };
                                        var det_classes = new InferRequestedOutputTensor() { Name = "detection_classes" };

                                        var det_multiclass_scores = new InferRequestedOutputTensor() { Name = "detection_multiclass_scores" };
                                        var det_anchor_indices = new InferRequestedOutputTensor() { Name = "detection_anchor_indices" };
                                        var raw_detection_boxes = new InferRequestedOutputTensor() { Name = "raw_detection_boxes" };
                                        var raw_detection_scores = new InferRequestedOutputTensor() { Name = "raw_detection_scores" };*/

                    var inferRequest = new ModelInferRequest()
                    {
                        
                        ModelName = modelInfo.Name,
                        ModelVersion = modelInfo.Versions.Last(),
                        Inputs = {input},
                        RawInputContents = { ByteString.CopyFrom(inputData) },
                        Outputs = {outputs}
    /*                    Outputs = {
                                    num_detections,
                                    det_boxes,
                                    det_scores,
                                    det_classes,
                                    det_multiclass_scores,
                                    det_anchor_indices,
                                    raw_detection_boxes,
                                    raw_detection_scores},*/
                      


                    };


                    var response = await client.ModelInferAsync(inferRequest);

                    return response;

                }catch (Exception ex)
                {
                    throw ex;
                }
            }
            else
            {
                throw new Exception("Channel not ready.");
            }
        }

        public void Dispose()
        {
            if (channel != null && channel.State == ChannelState.Ready)
            {
                channel.ShutdownAsync().Wait();
                channel = null;
                client = null;
            }
        }
    }
}
