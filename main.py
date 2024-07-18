import torch
import whisper_timestamped as whisper
from pydub import AudioSegment
import pysrt, os, json
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip

MAX_FILE_SIZE_MB = 25
OVERLAP_MS = 0


class VideoSubtitleProcessor:
    def __init__(self, video_path, audio_path=None, output_path=None):
        if not video_path:
            raise ValueError("Video path must be provided.")
        if not output_path:
            self.output_path = "./video_subtitle"
        else:
            self.output_path = output_path
        self.temp_dir = os.path.join(self.output_path, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.audio_path = audio_path
        self.video_path = os.path.abspath(video_path)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model("large-v3", device=device)
        self.is_has_audio = False if audio_path else True

    def _load_and_process_audio(self):
        if self.audio_path is None:
            self.audio_path = self._extract_audio_from_video()

        def split_audio(audio: AudioSegment, max_size_mb: int, overlap_ms: int, file_size: int):
            max_size_bytes = max_size_mb * 1024 * 1024
            segments = []
            start_ms = 0
            audio_length_ms = audio.duration_seconds * 1000
            print(f"Audio length: {audio_length_ms} ms")
            print(f"Audio size: {file_size} bytes")
            # 计算每毫秒的字节数
            bytes_per_ms = file_size / audio_length_ms

            while start_ms < audio_length_ms:
                # 计算当前片段的结束时间（毫秒）
                end_ms = start_ms + (max_size_bytes / bytes_per_ms)
                end_ms = min(end_ms, audio_length_ms)

                # 提取当前片段
                segment = audio[start_ms:end_ms]
                segments.append(segment)

                if end_ms == audio_length_ms:
                    break

                # 更新下一个片段的起始时间，考虑重叠
                start_ms = end_ms - overlap_ms

            return segments

        audio = AudioSegment.from_file(self.audio_path, format="mp4")
        file_size = os.path.getsize(self.audio_path)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            audio_segments = split_audio(audio, MAX_FILE_SIZE_MB, OVERLAP_MS, file_size=file_size)
        else:
            audio_segments = [audio]
        print(f"{len(audio_segments)} audio segments extracted.")
        for i, audio_segment in enumerate(audio_segments):
            audio_segment.export(f"{self.temp_dir}/temp_{i+1}.wav", format="wav")

        return

    def _extract_audio_from_video(self):
        audio_path = os.path.join(self.temp_dir, "audio.wav")
        video = VideoFileClip(self.video_path)
        audio = video.audio
        audio.write_audiofile(audio_path)
        video.close()
        return audio_path

    def _generate_subtitles(self):
        files = []
        for file in os.listdir(self.temp_dir):
            if file.endswith(".wav") and file.startswith("temp_"):
                files.append(os.path.join(self.temp_dir, file))

        def process_file(file):
            audio = whisper.load_audio(file)
            file_name = os.path.basename(file).split(".")[0].split("_")[-1]
            result = whisper.transcribe(self.model, audio, language="en")
            with open(f"{self.temp_dir}/result_{file_name}.json", "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            subs = pysrt.SubRipFile()
            for i, chunk in enumerate(result["segments"]):
                start_time = chunk["start"] * 1000
                end_time = chunk["end"] * 1000
                subs.append(pysrt.SubRipItem(index=i + 1, start=pysrt.SubRipTime(milliseconds=int(start_time)), end=pysrt.SubRipTime(milliseconds=int(end_time)), text=chunk["text"]))
            subs.save(f"{self.temp_dir}/subtitles_{file_name}.srt", encoding="utf-8")

        # import concurrent.futures
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     future_to_file = {executor.submit(process_file, idx, file): file for idx, file in enumerate(files)}
        #     for future in concurrent.futures.as_completed(future_to_file):
        #         file = future_to_file[future]
        #         try:
        #             future.result()
        #         except Exception as e:
        #             print(f"Unexpected error with file {file}: {str(e)}")

        for file in files:
            process_file(file)

        # 合并字幕，并去掉重叠部分
        combined_subs = pysrt.SubRipFile()
        file_paths = []
        last_end_time = 0
        for file in os.listdir(self.temp_dir):
            if file.startswith("subtitles_") and file.endswith(".srt"):
                file_paths.append(os.path.join(self.temp_dir, file))
        file_paths = sorted(file_paths, key=lambda x: int(x.split(".")[1].split("_")[-1]))
        for file_path in file_paths:
            subs = pysrt.open(file_path)
            for sub in subs:
                new_start_ms = last_end_time + sub.start.ordinal
                new_end_ms = last_end_time + sub.end.ordinal
                new_sub = pysrt.SubRipItem(index=len(combined_subs) + 1, start=pysrt.SubRipTime(milliseconds=new_start_ms), end=pysrt.SubRipTime(milliseconds=new_end_ms), text=sub.text)
                combined_subs.append(new_sub)
            last_end_time += subs[-1].end.ordinal

        combined_subs.save(os.path.join(self.output_path, "subtitles.srt"), encoding="utf-8")

        return os.path.join(self.output_path, "subtitles.srt")

    def _create_subtitle_clip(self, video_size):
        file_path = f"{self.output_path}/subtitles.srt"
        subs = pysrt.open(file_path)
        subtitle_clips = []
        for sub in subs:
            start = sub.start.ordinal / 1000
            end = sub.end.ordinal / 1000
            duration = end - start
            max_width = video_size[0] - 100
            text_clip = TextClip(sub.text, fontsize=20, color="black", font="Arial", method="caption", size=(max_width, None), align="center")
            subtitle_y_position = video_size[1] - 50 - text_clip.size[1]
            text_clip = text_clip.set_position(("center", subtitle_y_position)).set_duration(duration).set_start(start)
            subtitle_clips.append(text_clip)
        return subtitle_clips

    def _combine_video_and_subtitles(self):
        video = VideoFileClip(self.video_path)
        if not self.is_has_audio:
            audio = AudioFileClip(self.audio_path)
            video = video.set_audio(audio)

        subtitle_clips = self._create_subtitle_clip(video.size)
        final_video = CompositeVideoClip([video] + subtitle_clips, size=video.size)
        final_video.write_videofile(f"{self.output_path}/subtitle_video.mp4", codec="libx264", audio_codec="aac")
        video.close()
        final_video.close()

    def process_video(self):
        self._load_and_process_audio()
        _ = self._generate_subtitles()
        self._combine_video_and_subtitles()

        # video = VideoFileClip(self.video_path).subclip(0, self.duration_limit)
        # final_duration = max(video.duration, self.duration_limit)

        # subs = pysrt.open(subtitles_path)
        # subtitle_clips = [self._create_subtitle_clip(sub, video.size) for sub in subs if sub.end.ordinal / 1000 <= final_duration]

        # final_video = CompositeVideoClip([video] + subtitle_clips, size=video.size)
        # final_video = final_video.set_duration(final_duration).set_audio(AudioFileClip(audio_file).set_duration(final_duration))

        # final_video.write_videofile(self.output_path, codec="libx264", audio_codec="aac")

        # video.close()
        # final_video.close()

        print("Video processing completed.")


if __name__ == "__main__":
    # Example usage:
    processor = VideoSubtitleProcessor(video_path="dis_sys_1.mp4")
    processor.process_video()
