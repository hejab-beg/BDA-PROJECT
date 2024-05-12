import os
import subprocess

def compress_audio_ffmpeg(input_folder, target_bitrate=64):
    # Iterate through all files in the input folder
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".mp3"):
                input_path = os.path.join(root, filename)
                
                # Generate temporary output path
                temp_output_path = input_path.replace(".mp3", "_temp.mp3")
                
                # FFmpeg command for compressing audio without changing metadata
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", input_path,
                    "-b:a", f"{target_bitrate}k",
                    "-map_metadata", "0",
                    "-id3v2_version", "3",
                    "-write_id3v1", "1",
                    "-y",  # Overwrite output files without asking
                    temp_output_path
                ]
                
                # Execute FFmpeg command
                subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                # Replace original file with the compressed one
                os.replace(temp_output_path, input_path)
                
                print(f"Compressed {filename} successfully.")

if __name__ == "__main__":
    input_folder = "test_dataset"
    target_bitrate = 64  # Set the target bitrate in kbps

    # Compress audio files in the input folder
    print("Starting compression...")
    compress_audio_ffmpeg(input_folder, target_bitrate)
    print("Compression completed.")