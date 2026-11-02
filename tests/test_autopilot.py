# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Autopilot: persona, kalendarz zapasu, wybor modelki, planowanie sesji, kolejnosc w kolejce."""
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import autopilot, content, history
from app import queue_manager as qm

TODAY = dt.date(2026, 11, 1)
DAY = [(TODAY + dt.timedelta(days=d)).isoformat() for d in range(5)]


class PersonaTests(unittest.TestCase):
    def test_values_are_kept_within_limits(self):
        p = autopilot.clean({"age": 17, "photos_per_day": 99, "video_seconds": 1,
                             "adult_share": 0.5, "buffer_days": 0,
                             "wardrobe": "biała lniana sukienka\n\n  czarne jeansy  \n"})
        self.assertEqual(p["age"], 21)  # nigdy ponizej 21
        self.assertEqual((p["photos_per_day"], p["video_seconds"], p["buffer_days"]), (20, 2, 1))
        self.assertNotIn("adult_share", p)  # dawny udzial sesji 18+ odpada
        self.assertEqual(p["wardrobe"], ["biała lniana sukienka", "czarne jeansy"])

    def test_saved_persona_is_read_back(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(autopilot.characters, "DIR", Path(tmp)):
            (Path(tmp) / "mia").mkdir()
            autopilot.save_persona("mia", {"age": 26, "places": "Lizbona\nTulum",
                                           "enabled": True})
            p = autopilot.persona("mia")
        self.assertEqual((p["age"], p["places"], p["enabled"]), (26, ["Lizbona", "Tulum"], True))
        self.assertEqual(p["photos_per_day"], autopilot.DEFAULTS["photos_per_day"])

    def test_persona_saved_with_an_18_plus_share_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(autopilot.characters, "DIR", Path(tmp)):
            (Path(tmp) / "mia").mkdir()
            (Path(tmp) / "mia" / "persona.json").write_text(
                json.dumps({"age": 24, "adult_share": 0.1, "enabled": True}), encoding="utf-8")
            p = autopilot.persona("mia")
            saved = json.loads(json.dumps(autopilot.save_persona("mia", p)))
        self.assertTrue(p["enabled"])
        self.assertNotIn("adult_share", p)
        self.assertNotIn("adult_share", saved)


class CalendarTests(unittest.TestCase):
    P = autopilot.clean({"photos_per_day": 2, "videos_per_day": 1, "buffer_days": 3})

    def test_legacy_or_unverified_content_cannot_fill_the_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "existing.png"
            image.write_bytes(b"image")
            base = {"autopilot": True, "profile": "mia", "day": DAY[0],
                    "mode": "txt2img", "files": [str(image)]}
            entries = [base, {**base, "qc": {"ok": True}},
                       {**base, "qc": {"safe": True}, "level": content.LEGACY_ADULT},
                       {**base, "qc": {"safe": True, "ok": False}}]
            counts = autopilot.day_counts("mia", entries, [], queued=False)
        self.assertEqual(counts, {DAY[0]: {"image": 1, "video": 0}})

    def test_counts_ready_and_queued_content_per_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = [str(Path(tmp) / f"{i}.png") for i in range(4)]
            for p in f[:3]:
                Path(p).write_bytes(b"x")
            entries = [
                {"autopilot": True, "profile": "mia", "day": DAY[0], "mode": "txt2img", "files": [f[0]], "qc": {"ok": True, "safe": True}},
                {"autopilot": True, "profile": "mia", "day": DAY[0], "mode": "video_i2v", "files": [f[1]], "qc": {"ok": True, "safe": True}},
                {"autopilot": True, "profile": "mia", "day": DAY[0], "mode": "txt2img", "files": [f[2]],
                 "qc": {"ok": False}},                               # nie przeszlo kontroli
                {"autopilot": True, "profile": "mia", "day": DAY[1], "mode": "txt2img", "files": [f[3]]},  # brak pliku
                {"profile": "mia", "mode": "txt2img", "files": [f[0]]},                    # reczne
                {"autopilot": True, "profile": "nova", "day": DAY[0], "mode": "txt2img", "files": [f[0]]}]
            jobs = [qm.Job(id=1, kind="obraz", label="", params={"autopilot": True,
                                                                  "character": "mia", "day": DAY[0]}),
                    qm.Job(id=2, kind="wideo", label="", params={"autopilot": True,
                                                                  "character": "mia", "day": DAY[1]})]
            counts = autopilot.day_counts("mia", entries, jobs)
            ready = autopilot.day_counts("mia", entries, jobs, queued=False)
        self.assertEqual(counts, {DAY[0]: {"image": 2, "video": 1}, DAY[1]: {"image": 0, "video": 1}})
        self.assertEqual(ready, {DAY[0]: {"image": 1, "video": 1}})

    def test_buffer_and_first_gap(self):
        counts = {DAY[0]: {"image": 2, "video": 1}, DAY[1]: {"image": 1, "video": 1}}
        self.assertEqual(autopilot.buffer_days(self.P, counts, TODAY), 1)
        self.assertEqual(autopilot.gap(self.P, counts, TODAY), (DAY[1], 1, 0))
        full = {d: {"image": 2, "video": 1} for d in DAY[:3]}
        self.assertIsNone(autopilot.gap(self.P, full, TODAY))  # okno 3 dni pelne

    def test_session_theme_is_always_sfw(self):
        p = autopilot.clean({})
        dressed = (True, json.dumps({"theme": "Lizbona, taras o zachodzie, biała sukienka"}))
        with patch.object(autopilot.llm, "ask", return_value=dressed) as ask:
            ok, theme = autopilot.session_theme(p, DAY[0], [])
        self.assertTrue(ok)
        self.assertIn(content.RULE, ask.call_args.args[0])
        nude = (True, json.dumps({"theme": "sesja nago w sypialni o poranku"}))
        with patch.object(autopilot.llm, "ask", return_value=nude):
            ok, msg = autopilot.session_theme(p, DAY[0], [])
        self.assertFalse(ok)  # temat z nagoscia nie trafia do kolejki
        self.assertIn("SFW", msg)


class TickTests(unittest.TestCase):
    CHARS = [{"slug": "mia", "name": "Mia", "gender": "woman"},
             {"slug": "nova", "name": "Nova", "gender": "woman"}]

    def run_tick(self, counts, enabled=True, paused=False, jobs=()):
        persona = autopilot.clean({"photos_per_day": 2, "videos_per_day": 0, "buffer_days": 3,
                                   "enabled": True})
        with patch.object(autopilot, "enabled", return_value=enabled), \
                patch.object(qm, "paused", return_value=paused), \
                patch.object(qm, "snapshot", return_value=list(jobs)), \
                patch.object(autopilot.characters, "all_characters", return_value=self.CHARS), \
                patch.object(autopilot.characters, "usable", return_value=True), \
                patch.object(autopilot, "persona", return_value=persona), \
                patch.object(autopilot, "day_counts",
                             side_effect=lambda slug, *a, **k: counts[slug]), \
                patch.object(autopilot.packs, "day_done", return_value=True), \
                patch.object(autopilot, "plan_session",
                             side_effect=lambda ch, p, day, ph, vi: (True, f"{ch['slug']} {day} {ph}")) as plan:
            return autopilot.tick(TODAY), plan

    def test_a_complete_day_gets_its_pack_before_new_sessions(self):
        persona = autopilot.clean({"photos_per_day": 2, "videos_per_day": 0, "buffer_days": 3,
                                   "enabled": True})
        counts = {DAY[0]: {"image": 2, "video": 0}}
        with patch.object(autopilot, "enabled", return_value=True), \
                patch.object(qm, "paused", return_value=False), \
                patch.object(qm, "snapshot", return_value=[]), \
                patch.object(autopilot.characters, "all_characters", return_value=self.CHARS[:1]), \
                patch.object(autopilot.characters, "usable", return_value=True), \
                patch.object(autopilot, "persona", return_value=persona), \
                patch.object(autopilot, "day_counts", return_value=counts), \
                patch.object(autopilot.packs, "day_done", return_value=False), \
                patch.object(autopilot.packs, "build_day", return_value=(True, "📦 paczka")) as build, \
                patch.object(autopilot, "plan_session") as plan:
            msg = autopilot.tick(TODAY)
        self.assertEqual(msg, "📦 paczka")
        build.assert_called_once_with("mia", DAY[0])
        plan.assert_not_called()  # sesja w nastepnym obiegu

    def test_the_model_with_the_shortest_buffer_goes_first(self):
        counts = {"mia": {DAY[0]: {"image": 2}}, "nova": {}}
        msg, plan = self.run_tick(counts)
        self.assertEqual(msg, f"nova {DAY[0]} 2")
        counts = {"mia": {DAY[0]: {"image": 2}}, "nova": {DAY[0]: {"image": 2}, DAY[1]: {"image": 1}}}
        msg, _ = self.run_tick(counts)
        self.assertEqual(msg, f"mia {DAY[1]} 2")

    def test_nothing_is_planned_when_off_paused_busy_or_full(self):
        full = {s: {d: {"image": 2} for d in DAY} for s in ("mia", "nova")}
        self.assertEqual(self.run_tick({}, enabled=False)[0], "disabled")
        self.assertEqual(self.run_tick({}, paused=True)[0], "paused")
        busy = [SimpleNamespace(status=qm.PENDING, params={}, kind="obraz")]
        self.assertTrue(self.run_tick({}, jobs=busy)[0].startswith("queue busy"))
        msg, plan = self.run_tick(full)
        self.assertEqual(msg, "content buffer full")
        plan.assert_not_called()


class PlanSessionTests(unittest.TestCase):
    def test_identity_reaches_theme_and_both_prompt_kinds(self):
        ch = {"slug": "alex", "name": "Alex", "gender": "man", "look": "short curly hair"}
        p = autopilot.clean({"age": 25})
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(autopilot.characters, "DIR", Path(tmp)), \
                patch.object(autopilot, "session_theme", return_value=(True, "A city walk")) as theme, \
                patch.object(autopilot.planner, "write_series", return_value=["A city walk"]) as write, \
                patch.object(autopilot.planner, "enqueue", return_value=(2, 0)):
            (Path(tmp) / "alex").mkdir()
            ok, _ = autopilot.plan_session(ch, p, DAY[0], 1, 1)
        self.assertTrue(ok)
        self.assertIs(theme.call_args.kwargs["character"], ch)
        self.assertEqual([call.args[0] for call in write.call_args_list], ["image", "video"])
        for call in write.call_args_list:
            self.assertIn("exactly 25 years old", call.args[6])
            self.assertIn("gender: man", call.args[6])
            self.assertIn("short curly hair", call.args[6])

    def test_session_theme_prompt_keeps_persona_age_and_character_gender(self):
        p = autopilot.clean({"age": 25})
        with patch.object(autopilot.llm, "ask", return_value=(True, '{"theme": "A city walk"}')) as ask:
            ok, _ = autopilot.session_theme(p, DAY[0], [], character={"gender": "man"})
        self.assertTrue(ok)
        system, user = ask.call_args.args
        self.assertIn("exact age", system)
        self.assertNotIn("woman aged", system)
        self.assertIn("exactly 25 years old; gender: man", user)

    def test_session_goes_to_the_queue_for_that_model_and_day(self):
        ch = {"slug": "mia", "name": "Mia", "gender": "woman"}
        p = autopilot.clean({"photos_per_day": 2, "videos_per_day": 1, "video_seconds": 6})
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(autopilot.characters, "DIR", Path(tmp)), \
                patch.object(autopilot, "session_theme",
                             return_value=(True, "Lizbona, taras o zachodzie, biała sukienka")), \
                patch.object(autopilot.planner, "write_series",
                             side_effect=lambda kind, theme, n, *a: [f"{kind} prompt {i}" for i in range(n)]), \
                patch.object(autopilot.planner, "enqueue", return_value=(3, 0)) as enqueue:
            (Path(tmp) / "mia").mkdir()
            ok, msg = autopilot.plan_session(ch, p, DAY[1], 2, 1)
            log = (Path(tmp) / "mia" / "sessions.jsonl").read_text(encoding="utf-8")
            themes = autopilot.recent_themes("mia")
        self.assertTrue(ok)
        rows, refs, order = enqueue.call_args.args
        kw = enqueue.call_args.kwargs
        self.assertEqual([r[0] for r in rows], ["Photo", "Photo", "Video"])
        self.assertEqual(rows[2][2], 6)  # dlugosc filmu z persony
        self.assertIs(kw["character"], ch)
        self.assertEqual(kw["speed"], autopilot.video_gen.QUALITY)  # tryb filmow z persony
        self.assertEqual((kw["extra_params"]["autopilot"], kw["extra_params"]["day"]), (True, DAY[1]))
        self.assertEqual(json.loads(log)["theme"], "Lizbona, taras o zachodzie, biała sukienka")
        self.assertNotIn("level", json.loads(log))
        self.assertEqual(themes, ["Lizbona, taras o zachodzie, biała sukienka"])
        self.assertTrue(all(len(r) == 5 for r in rows))  # bez dawnej kolumny "Treść"


class QueueOrderTests(unittest.TestCase):
    def test_manual_jobs_run_before_autopilot_jobs(self):
        jobs = [qm.Job(id=1, kind="obraz", label="auto", params={"autopilot": True}),
                qm.Job(id=2, kind="obraz", label="reczne", params={}),
                qm.Job(id=3, kind="obraz", label="auto2", params={"autopilot": True})]
        with patch.object(qm, "_JOBS", jobs), \
                patch.object(qm, "_STATE_INFO", {"next_id": 4, "paused": False}):
            self.assertEqual(qm._next().id, 2)
            jobs[1].status = qm.DONE
            self.assertEqual(qm._next().id, 1)

    def test_autopilot_day_and_session_reach_the_history(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(history, "_HISTORY_FILE", Path(tmp) / "history.jsonl"), \
                patch.object(history.config, "HISTORY_DIR", Path(tmp)):
            history.job_context({"autopilot": True, "day": DAY[0], "session": "s1",
                                 "character": "mia"})
            entry = history.record({"files": ["a.png"]})
            history.job_context(None)
        self.assertEqual((entry["autopilot"], entry["day"], entry["session"], entry["profile"]),
                         (True, DAY[0], "s1", "mia"))


if __name__ == "__main__":
    unittest.main()
