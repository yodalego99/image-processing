using System;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace ImageProcessingFinal.Views;


public class VideoCaptureInfo
{
    public VideoCapture Video { get; set; }                 // Emgu.Cv.VideoCapture object (it can be a camera input or video input)
    public string? FilePath { get; set; }                   // Video file path
    public long? TotalDuration { get; set; }                // Length of video input in ms (milliseconds)
    public double? FPS { get; set; }                        // FPS (frames per second) of the video
    public double? DeltaFrameTime { get; set; }             // Time elapsed between frames in ms (milliseconds)
    private bool? IsWebcam { get; set; }                    // Webcamera input?

    public VideoCaptureInfo(VideoCapture Video, bool IsWebCam, string FilePath)
    {
        this.Video = Video;
        this.IsWebcam = IsWebCam;
        this.FPS = Convert.ToInt32(Video.Get(CapProp.Fps));
        this.DeltaFrameTime = 1000.0 / this.FPS;
        if (!IsWebCam)
        {
            this.TotalDuration = Convert.ToInt64(DeltaFrameTime * Convert.ToDouble(Video.Get(CapProp.FrameCount)));
            this.FilePath = FilePath;
        }
    }
}