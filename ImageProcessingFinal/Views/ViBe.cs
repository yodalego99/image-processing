using System;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace ImageProcessingFinal.Views;



public class ViBe
{
    // The fidelity of the background model
    int _n = 20;

    // Distance between two pixels colour in color space
    int _r = 20;

    // Required matches to be added to the background model
    int _bgMMin = 2;

    // Rate of decay - bigger values tend to cause ghosting
    int _phi = 16;
    
    int _frameWidth, _frameHeight;
    Image<Rgba, byte> _frameImage;

    // Background model
    byte[,,,] _samples;
    // Segmentation map - result of the ViBe background removal operation
    private Image<Rgba, byte> _segMap;
    byte[,,] _frameImageBytes;
    byte[,,] _segMapBytes;
    Mat _frameRead;
    
    double _frameDifferencePercentage = 0.125; // This is the difference between the
                                              // previous and current frame if the
                                              // difference is big then we
                                              // reinitialize the background model
    int _matchCount; // Number of matches
    bool _shakyCamera = false;   // This indicates, whether the camera shaking detection is on or off
    byte[,,,] _compareFrames; // Two consequent frames
    private void BackgroundModelInitialization()
    {
        for (int k = 0; k < _n; k++)
        {
            Parallel.For(0, ProcessorCount, cpuCoreId =>
            {
                var max = _frameWidth * (cpuCoreId + 1) / ProcessorCount;
                for (int x = _frameWidth * cpuCoreId / ProcessorCount; x < max; x++)
                {
                    for (int y = 0; y < _frameHeight; y++)
                    {
                        if (_shakyCamera)
                        {
                            _compareFrames[x, y, 0, 0] = _frameImageBytes[y, x, 0];
                            _compareFrames[x, y, 0, 1] = _frameImageBytes[y, x, 1];
                            _compareFrames[x, y, 0, 2] = _frameImageBytes[y, x, 2];
                        }

                        _samples[x, y, k, 0] = _frameImageBytes[y, x, 0];
                        _samples[x, y, k, 1] = _frameImageBytes[y, x, 1];
                        _samples[x, y, k, 2] = _frameImageBytes[y, x, 2];
                    }
                }
            });
        }
    }

    private void BackgroundModelUpdate(int i)
    {
        Parallel.For(0, ProcessorCount, cpuCoreId =>
        {
            var max = _frameWidth * (cpuCoreId + 1) / ProcessorCount;
            for (int x = _frameWidth * cpuCoreId / ProcessorCount; x < max; x++)
            {
                for (int y = 0; y < _frameHeight; y++)
                {
                    int count = 0;
                    int index = 0;
                    int db, dg, dr = 0;
                    if (i % 2 == 0 && i != 0 && _shakyCamera)
                    {
                        _compareFrames[x, y, 0, 0] = _frameImageBytes[y, x, 0];
                        _compareFrames[x, y, 0, 1] = _frameImageBytes[y, x, 1];
                        _compareFrames[x, y, 0, 2] = _frameImageBytes[y, x, 2];
                        if ((0.11d * _compareFrames[x, y, 0, 0] + 0.59d * _compareFrames[x, y, 0, 1] +
                             0.3d * _compareFrames[x, y, 0, 2]) == (0.11d * _compareFrames[x, y, 1, 0] +
                                                                   0.59d * _compareFrames[x, y, 1, 1] +
                                                                   0.3d * _compareFrames[x, y, 1, 2]))
                        {
                            _matchCount++;
                        }
                    }
                    else if (i % 2 == 1 && _shakyCamera)
                    {
                        _compareFrames[x, y, 1, 0] = _frameImageBytes[y, x, 0];
                        _compareFrames[x, y, 1, 1] = _frameImageBytes[y, x, 1];
                        _compareFrames[x, y, 1, 2] = _frameImageBytes[y, x, 2];
                        if ((0.11d * _compareFrames[x, y, 0, 0] + 0.59d * _compareFrames[x, y, 0, 1] +
                             0.3d * _compareFrames[x, y, 0, 2]) == (0.11d * _compareFrames[x, y, 1, 0] +
                                                                   0.59d * _compareFrames[x, y, 1, 1] +
                                                                   0.3d * _compareFrames[x, y, 1, 2]))
                        {
                            _matchCount++;
                        }
                    }

                    while ((count < _bgMMin) && (index < _n))
                    {
                        db = (int)Math.Abs(_frameImageBytes[y, x, 0] - _samples[x, y, index, 0]);
                        dg = (int)Math.Abs(_frameImageBytes[y, x, 1] - _samples[x, y, index, 1]);
                        dr = (int)Math.Abs(_frameImageBytes[y, x, 2] - _samples[x, y, index, 2]);
                        if (db < _r && dg < _r && dr < _r)
                        {
                            count++;
                        }

                        index++;
                    }

                    if (count >= _bgMMin)
                    {
                        if (OnlyBackground)
                        {
                            _segMapBytes[y, x, 0] = _frameImageBytes[y, x, 0];
                            _segMapBytes[y, x, 1] = _frameImageBytes[y, x, 1];
                            _segMapBytes[y, x, 2] = _frameImageBytes[y, x, 2];
                        }
                        else
                        {
                            _segMapBytes[y, x, 0] = byte.MinValue;
                            _segMapBytes[y, x, 1] = byte.MinValue;
                            _segMapBytes[y, x, 2] = byte.MinValue;
                        }

                        int rand = _rnd.Next(0, _phi - 1);
                        if (rand == 0)
                        {
                            rand = _rnd.Next(0, _n - 1);
                            _samples[x, y, rand, 0] = _frameImageBytes[y, x, 0];
                            _samples[x, y, rand, 1] = _frameImageBytes[y, x, 1];
                            _samples[x, y, rand, 2] = _frameImageBytes[y, x, 2];
                        }

                        rand = _rnd.Next(0, _phi - 1);
                        if (rand == 0)
                        {
                            int xNg, yNg;
                            rand = _rnd.Next(0, _n - 1);
                            xNg = GetRandomNeighbourPixel(x);
                            yNg = GetRandomNeighbourPixel(y);
                            _samples[xNg, yNg, rand, 0] = _frameImageBytes[y, x, 0];
                            _samples[xNg, yNg, rand, 1] = _frameImageBytes[y, x, 1];
                            _samples[xNg, yNg, rand, 2] = _frameImageBytes[y, x, 2];
                        }
                    }
                    else
                    {
                        if (OnlyForeground)
                        {
                            _segMapBytes[y, x, 0] = _frameImageBytes[y, x, 0];
                            _segMapBytes[y, x, 1] = _frameImageBytes[y, x, 1];
                            _segMapBytes[y, x, 2] = _frameImageBytes[y, x, 2];
                        }
                        else if (OnlyBackground)
                        {
                            if ((x + y) % 2 == 0)
                            {
                                _segMapBytes[y, x, 0] = byte.MaxValue;
                                _segMapBytes[y, x, 1] = byte.MinValue;
                                _segMapBytes[y, x, 2] = byte.MaxValue;
                            }
                            else
                            {
                                _segMapBytes[y, x, 0] = byte.MinValue;
                                _segMapBytes[y, x, 1] = byte.MinValue;
                                _segMapBytes[y, x, 2] = byte.MinValue;
                            }
                        }
                        else
                        {
                            _segMapBytes[y, x, 0] = byte.MaxValue;
                            _segMapBytes[y, x, 1] = byte.MaxValue;
                            _segMapBytes[y, x, 2] = byte.MaxValue;
                        }
                    }
                }
            }
        });
        if ((double)(_matchCount) / (double)(_frameWidth * _frameHeight) < _frameDifferencePercentage && _shakyCamera)
        {
            BackgroundModelInitialization();
        }

        _matchCount = 0;
    }
    private int GetRandomNeighbourPixel(int coord)
    {
        int[] var = [-1, 0, 1];

        var rnd = new Random();

        if (coord == (_frameHeight - 1) || (coord == _frameWidth - 1) || coord == 0)
        {
            return coord;
        }
        else
        {
            return coord + var[rnd.Next(3)];
        }
    }
}