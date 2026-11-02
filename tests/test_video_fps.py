# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from app import ffmpeg_utils, video_gen

ROOT = Path(__file__).resolve().parents[1]
# Contsy's change to the RIFE node, installed by app/setup_windows.py (see patches/)
PLAN = ROOT / 'patches/ComfyUI-Frame-Interpolation/vfi_models/rife/frame_rate.py'
spec = importlib.util.spec_from_file_location('rife_frame_rate', PLAN)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


class FrameRatePlanTests(unittest.TestCase):
    def test_wan_uses_exact_thirtieths_without_dropping_intermediate_frames(self):
        tasks, layout = planner.frame_rate_plan(81, 16, 30)
        self.assertEqual(len(layout), 151)  # 5 seconds plus its endpoint
        per_pair = {}
        for pair, fraction in tasks:
            per_pair.setdefault(pair, []).append(fraction)
        for index, (pair, slot) in enumerate(layout):
            fraction = 0 if slot < 0 else per_pair[pair][slot]
            self.assertAlmostEqual((pair + fraction) / 16, index / 30)

    def test_integer_ratio_reproduces_existing_multiplier_timestamps(self):
        tasks, layout = planner.frame_rate_plan(17, 16, 48)
        self.assertEqual(tasks, [(i, t / 3) for i in range(16) for t in (1, 2)])
        self.assertEqual(len(layout), 49)

    def test_lipsync_preserves_last_source_frame_and_does_not_extrapolate(self):
        tasks, layout = planner.frame_rate_plan(25, 25, 30)
        self.assertEqual(len(layout), 30)
        self.assertEqual(layout[-1], (24, -1))
        self.assertTrue(all(0 < fraction < 1 for _, fraction in tasks))
        self.assertTrue(all(pair < 24 for pair, _ in tasks))

    def test_single_frame(self):
        self.assertEqual(planner.frame_rate_plan(1, 25, 30), ([], [(0, -1)]))

    def test_lipsync_uses_same_full_precision_rife_at_30_fps(self):
        graph = video_gen._lipsync_fps_workflow('test.mp4')
        rife = graph['50']['inputs']
        self.assertEqual((rife['source_fps'], rife['target_fps']), (25, 30))
        self.assertEqual((rife['ckpt_name'], rife['dtype']), ('rife49.pth', 'float32'))
        self.assertTrue(rife['ensemble'])
        self.assertFalse(rife['fast_mode'])
        self.assertEqual(graph['90']['inputs']['frame_rate'], 30)
        self.assertEqual(graph['90']['inputs']['crf'], 0)

    def test_lipsync_conversion_uses_original_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'source.mp4', Path(folder) / 'out.mp4'
            client = Mock()
            client.run_and_wait.return_value.files = [{'filename': 'aistudio_lipsync_rife_1.mp4'}]
            client.download_output.return_value = b'test'
            with patch.object(ffmpeg_utils, 'probe_duration', return_value=2.0), \
                 patch.object(ffmpeg_utils, 'convert_fps', return_value=(True, str(output))) as convert:
                self.assertTrue(video_gen._lipsync_to_output_fps(client, source, output)[0])
                self.assertEqual(convert.call_args.args[2], 30)
                self.assertEqual(convert.call_args.kwargs['duration'], 2.0)
            self.assertFalse(output.with_name('out_rife.mp4').exists())


def _profile(path):
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=profile', '-of', 'json', str(path)]))
    return info['streams'][0]['profile']


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class VideoTimingTests(unittest.TestCase):
    def test_join_removes_shared_endpoint_and_keeps_requested_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            source, joined = Path(folder) / 'part.mp4', Path(folder) / 'joined.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'testsrc2=size=128x72:rate=30', '-frames:v', '31',
                            '-c:v', 'libx264', '-crf', '0', str(source)], check=True)
            ok, error = ffmpeg_utils.join_segments([str(source), str(source)], joined, 30,
                                                   [0, 0], duration=2.0)
            self.assertTrue(ok, error)
            self.assertEqual(ffmpeg_utils.video_fps(joined), 30)
            self.assertAlmostEqual(ffmpeg_utils.probe_duration(joined), 2.0, places=2)

    def test_conversion_and_audio_mux_keep_30_frames_and_one_second(self):
        with tempfile.TemporaryDirectory() as folder:
            source, converted, audio, final = [Path(folder) / name for name in
                                             ('source.mp4', '30fps.mp4', 'audio.wav', 'final.mp4')]
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'testsrc2=size=128x72:rate=50:duration=1', '-c:v', 'libx264',
                            '-crf', '0', str(source)], check=True)
            ok, error = ffmpeg_utils.convert_fps(source, converted, 30, duration=1.0)
            self.assertTrue(ok, error)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'sine=frequency=440:duration=1', str(audio)], check=True)
            ok, error = ffmpeg_utils.combine_video_audio(converted, audio, final)
            self.assertTrue(ok, error)
            self.assertEqual(ffmpeg_utils.video_fps(final), 30)
            self.assertAlmostEqual(ffmpeg_utils.probe_duration(final), 1.0, places=2)
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams',
                'v:0', '-show_entries', 'stream=nb_frames', '-of', 'json', str(final)]))
            self.assertEqual(int(info['streams'][0]['nb_frames']), 30)

    def test_compress_turns_the_lossless_intermediate_into_plain_h264(self):
        with tempfile.TemporaryDirectory() as folder:
            source, audio, final, silent = [Path(folder) / name for name in
                                            ('lossless.mp4', 'audio.wav', 'final.mp4', 'silent.mp4')]
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'testsrc2=size=128x72:rate=30:duration=1', '-c:v', 'libx264',
                            '-crf', '0', '-pix_fmt', 'yuv420p', str(source)], check=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'sine=frequency=440:duration=1', str(audio)], check=True)
            self.assertEqual(_profile(source), 'High 4:4:4 Predictive')  # jak plik z convert_fps
            for out, track in ((final, audio), (silent, None)):
                ok, error = ffmpeg_utils.combine_video_audio(source, track, out, compress=True)
                self.assertTrue(ok, error)
                self.assertEqual(_profile(out), 'High')
                self.assertEqual(ffmpeg_utils.video_fps(out), 30)
                self.assertAlmostEqual(ffmpeg_utils.probe_duration(out), 1.0, places=2)


if __name__ == '__main__':
    unittest.main()
