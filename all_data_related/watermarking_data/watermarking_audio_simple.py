import wave
import numpy as np

def watermark_audio_simple_text(audio_file, watermark_text, save_file):

    with wave.open(audio_file, 'rb') as audio:
        params = audio.getparams()
        n_channels, sampwidth, framerate, n_frames = params[:4]
        audio_data = audio.readframes(n_frames)

    audio_array = np.frombuffer(audio_data, dtype=np.uint8)
    watermark_bin = ''.join(format(ord(i), '08b') for i in watermark_text)
    watermark_index = 0

    for i in range(len(audio_array)):
        if watermark_index >= len(watermark_bin):
            break

        byte = audio_array[i]
        bit = int(watermark_bin[watermark_index])
        if bit:
            audio_array[i] = byte | 1
        else:
            audio_array[i] = byte & ~1

        watermark_index += 1

    with wave.open(save_file, 'wb') as output_audio:
        output_audio.setparams(params)
        output_audio.writeframes(audio_array.tobytes())

audio_file = 'audio.wav'
watermark_text = 'Watermark Text'
save_file = 'watermarked.wav'

watermark_audio_simple_text(audio_file, watermark_text, save_file)
