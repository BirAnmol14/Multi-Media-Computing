# README

## Requirements:
- Python 3+ (download: https://www.python.org/)
- TensorFlow (pip install tensorflow in command prompt)
- Keras (pip install keras)
- OpenCV (pip install opencv-python)
- NumPy (pip install numpy)
- Dahuffman (pip install dahuffman)
- ffmpeg (follow steps: https://www.wikihow.com/Install-FFmpeg-on-Windows  , https://extract.me  -- to convert 7z file to zip) 

## Steps: All these files are in the code folder
[NOTE: Running the files might give a warning (deprecation due to compatibility issues of TensorFlow 2.0 library and TensorFlow 1.0 code)]
1. Train the auto-encoder (Optional, since I pretrained the model, takes a lot of time)
	- run train.py
2. Encode your video file [Huffman Encoding]
	- run encoder.py
3. Decode your video file [Decoding and Stream Generation]
	- run decoder.py
4. Control the bit rate to match the compressed file [Generating a better quality lower bandwidth stream than plain H.264 video]
	- ffmpeg -i Football_decoded.mp4 -vcodec libx264 -y -b:v 3800k Football_final.mp4 (run this command in command prompt (set location of prompt to data folder first) )

## Output files explained: 
- Results Directory:
	- Images from various stages of execution

- Saved_Model directory:
	- decoder_weights: weights generated for decoder after training autoencoder [nupmy array file]
	- encoder_weights: weights generated for encoder after training autoencoder [nupmy array file]
	- huffman_codec: byte stream of Huffman encoding done by encoder.py

- Data directory:
	- Football_raw.mp4: raw mp4(1920*1080,30fps) footage
	- Football_compressed.mp4: Plain H.264 compressed video (1920*1080,30fps)
	- Football_predicted.mp4: Residual predicted by model for the video
	- Football_decoded.mp4: The framewise plain addition video of compressed and residual footage
	- Football_final.mp4: Bit rate adjusted final compressed video generated
	- Football_residual: the residual generated [nupmy array file]

