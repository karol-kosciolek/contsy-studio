# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Tylko SFW: straz od polecenia uzytkownika do promptu modelu, zestawu i treningu postaci."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import char_gen, characters, content, history, llm, planner, qc

# Prawdziwy prompt z planera (sesja w sypialni, 29.10): model 9B napisal "zakryta
# przescieradlem, without exposing..." - zwrot, ktory przywoluje to, czego ma nie byc.
SOFTENED = ("a 25-year-old woman lying on her side atop a plush white duvet in a private "
            "luxurious modern villa bedroom, soft morning sunlight flooding the room from a "
            "nearby window, her body covered by the crisp white sheets in a way that hints at "
            "form without exposing nipples or genitalia, her long hair fanned out on the pillow, "
            "shot with a 50mm lens for a natural portrait feel")
BIKINI = ("a 24-year-old woman in a red two-piece bikini walking along the white sand of a "
          "tropical beach at golden hour, turquoise water behind her, palm leaves casting soft "
          "shadows, relaxed confident expression, shot with an 85mm lens, shallow depth of field")
CHARACTER = {"trigger": "mia_aist", "gender": "woman", "look": "Tanned skin, long brown hair"}


class GuardTests(unittest.TestCase):
    def test_polish_and_english_requests_for_nudity_are_refused(self):
        for text in ["7 zdjęć nago w sypialni", "topless na jachcie", "sesja bez stanika",
                     "naga w łóżku", "akty w studiu", "18+ sesja", "treści dla dorosłych",
                     "she is nude on the bed", "naked in the shower", "nsfw photo"]:
            self.assertTrue(content.asks_nudity(text), text)
            self.assertEqual(content.check(text), content.REFUSAL, text)

    def test_swimwear_lingerie_and_lookalike_words_stay_allowed(self):
        for text in ["w czerwonym bikini na plaży", "seksowna bielizna", "w biustonoszu",
                     "nagle zaczyna padać", "nagroda dla zwycięzcy", "seksapil i pewność siebie",
                     "nude lipstick and nude heels", "szminka w kolorze nude", "sexy dress",
                     "the sexiest look of the season", BIKINI]:
            self.assertFalse(content.asks_nudity(text), text)
            self.assertEqual(content.check(text), "", text)

    def test_obfuscated_paraphrased_and_more_languages_are_refused(self):
        for text in ["n.u.d.e photo of a woman", "n u d e woman", "nud3 woman on a beach",
                     "p0rn scene", "nսde woman", "she takes off all her clothes",
                     "taking off her bikini", "removes her bra", "striptease in a club",
                     "strips naked", "see-through lingerie", "oben ohne Model",
                     "modelo pelada na praia", "голая женщина на пляже", "гола жінка на пляжі",
                     "девушка без одежды", "обнаженная модель", "секс на пляже"]:
            self.assertTrue(content.asks_nudity(text), text)

    def test_everyday_phrases_near_the_new_rules_stay_allowed(self):
        for text in ["the plane takes off at sunset", "she takes off her jacket",
                     "comic strip style", "a strip mall at night", "mechanic strips down the engine",
                     "wire strippers on a workbench", "see-through glass door",
                     "голые деревья зимой", "после гола болельщики радуются",
                     "сексуальная девушка в красном платье", "pelada de futebol na praia",
                     "U.S.A. flag on a pole", "N.A.S.A. rocket launch",
                     "a 1080p photo of a 5-star hotel lobby", "naked eye astronomy, starry sky"]:
            self.assertFalse(content.asks_nudity(text), text)

    def test_results_from_the_old_18_plus_mode_are_recognised(self):
        self.assertTrue(content.legacy_adult({"level": "nsfw"}))
        self.assertFalse(content.legacy_adult({"level": "sfw"}))
        self.assertFalse(content.legacy_adult({}))
        self.assertFalse(content.legacy_adult(None))


class EnforceTests(unittest.TestCase):
    def test_removes_priming_phrases_and_rejects_nudity(self):
        out = content.enforce(SOFTENED)
        self.assertNotIn("nipples", out)
        self.assertNotIn("without exposing", out)
        self.assertIn("her long hair fanned out", out)  # reszta opisu zostaje
        self.assertIsNone(content.enforce("a nude woman lying on the bed, morning light"))

    def test_faceless_shot_is_not_treated_as_hiding(self):
        text = "photo from behind without showing her face, sunset over the sea"
        self.assertEqual(content.enforce(text), text)


class CharacterPromptTests(unittest.TestCase):
    def test_names_a_concrete_outfit_that_fits_the_scene(self):
        # ogolne "fully clothed" przegrywalo z LoRA (test 30.10: 7 z 8 nago), nazwany stroj nie
        cases = {"she relaxes in a luxurious master suite with city views":
                 "a fitted black knit dress with long sleeves",
                 "she lies on top of the bed in soft morning light": "long-sleeve satin pajamas",
                 "she is swimming in an infinity pool at noon": "a black one-piece swimsuit",
                 "she sits by the fireplace in a mountain chalet": "a cozy knit sweater and jeans",
                 "close-up with natural skin texture, soft light":
                 "a fitted black knit dress with long sleeves"}  # "skin" to nie narty
        for text, outfit in cases.items():
            out = characters.positive(CHARACTER, text)
            self.assertIn(f"The woman is wearing {outfit}.", out, text)
            self.assertNotIn("fully clothed", out)

    def test_character_lora_always_runs_weaker(self):
        self.assertLess(content.LORA_STRENGTH, 1.0)
        self.assertEqual(characters.loras(CHARACTER | {"slug": "mia"}, content.LORA_STRENGTH)[0]
                         ["strength_model"], content.LORA_STRENGTH)

    def test_keeps_the_described_outfit(self):
        out = characters.positive(CHARACTER, BIKINI)
        self.assertNotIn("is wearing", out)
        self.assertTrue(out.startswith("Photo of mia_aist woman."))

    def test_old_prompt_never_gets_nudity_added(self):
        out = characters.positive(CHARACTER, SOFTENED)
        self.assertNotIn("nude", out)
        self.assertNotIn("without exposing", out)
        self.assertIn("is wearing long-sleeve satin pajamas", out)


class PlannerTests(unittest.TestCase):
    ORDER = "5 zdjęć w czerwonym bikini na plaży"

    def test_order_asking_for_nudity_is_refused_before_the_model(self):
        with patch.object(llm, "ask") as ask:
            self.assertEqual(planner.make_plan("5 zdjęć nago w sypialni"),
                             (False, content.REFUSAL))
            self.assertEqual(planner.start("3 zdjęcia topless na jachcie", [], True),
                             (False, content.REFUSAL))
        ask.assert_not_called()

    def test_series_writer_cleans_prompts_and_drops_nudity(self):
        reply = (True, json.dumps({"prompts": [SOFTENED]}))
        with patch.object(llm, "ask", return_value=reply) as ask:
            cleaned = planner.write_series("image", "w sypialni rano", 1, 0, 1)
        self.assertIn(content.RULE, ask.call_args.args[0])
        nude = (True, json.dumps({"prompts": ["a nude " + BIKINI]}))
        with patch.object(llm, "ask", return_value=nude):
            dropped = planner.write_series("image", "w bikini", 1, 0, 1)
        self.assertEqual(len(cleaned), 1)
        self.assertNotIn("without exposing", cleaned[0])
        self.assertEqual(dropped, [])

    def test_enqueue_keeps_the_order_and_skips_nudity(self):
        rows = [["zdjęcie", "Kwadrat 1:1", 0, "—", BIKINI],
                ["zdjęcie", "Portret 3:4", 0, "—", "SFW", BIKINI],  # wiersz z dawna kolumna Tresc
                ["zdjęcie", "Kwadrat 1:1", 0, "—", "a nude " + BIKINI]]
        with patch.object(planner.queue_manager, "add") as add, \
                patch.object(characters, "tag", return_value=""):
            added, skipped = planner.enqueue(rows, [], self.ORDER)
        self.assertEqual((added, skipped), (2, 1))
        params = [c.args[2] for c in add.call_args_list]
        labels = [c.args[1] for c in add.call_args_list]
        self.assertTrue(all(p["idea"] == self.ORDER and "level" not in p for p in params))
        self.assertTrue(all("SFW" not in label and "18+" not in label for label in labels))
        self.assertEqual(params[1]["format"], "Portrait 3:4")


class PromptAssistantTests(unittest.TestCase):
    def test_idea_asking_for_nudity_is_refused_before_the_model(self):
        with patch.object(llm, "ensure", return_value=None), patch.object(llm, "ask") as ask:
            ok, msg = llm.write_prompt("naga w łóżku rano", "image")
        self.assertEqual((ok, msg), (False, content.REFUSAL))
        ask.assert_not_called()

    def test_model_output_with_nudity_is_never_handed_back(self):
        with patch.object(llm, "ensure", return_value=None), \
                patch.object(llm, "ask", return_value=(True, "a nude " + BIKINI)):
            ok, msg = llm.write_prompt("w bikini na plaży", "image")
        self.assertFalse(ok)
        self.assertIn("SFW", msg)

    def test_priming_phrases_are_removed_and_the_rule_is_given(self):
        with patch.object(llm, "ensure", return_value=None), \
                patch.object(llm, "ask", return_value=(True, SOFTENED)) as ask:
            ok, text = llm.write_prompt("w sypialni rano", "image")
        self.assertTrue(ok)
        self.assertNotIn("without exposing", text)
        self.assertIn(content.RULE, ask.call_args.args[0])


class CharacterSetTests(unittest.TestCase):
    def test_set_generator_shoots_no_nudity_or_lingerie(self):
        for scene in char_gen.SCENES:
            self.assertFalse(content.asks_nudity(scene), scene)
            self.assertNotIn("lingerie", scene)

    def test_description_asking_for_nudity_is_refused(self):
        self.assertEqual(char_gen.check("a 24-year-old blonde, full nudity in studio poses"),
                         content.REFUSAL)
        self.assertEqual(char_gen.check("a 24-year-old woman with long blonde hair"), "")


class TrainingTests(unittest.TestCase):
    def test_nude_training_photos_are_left_out(self):
        """LoRA uczona na aktach rozbierala postac - trening bierze tylko zdjecia SFW."""
        captions = {"01": "white t-shirt, smiling, half body", "02": "topless on a bed",
                    "03": "black dress, standing", "04": "jeans and sweater, close-up",
                    "05": "red bikini on a beach", "06": "gray hoodie on a sofa",
                    "07": "blue suit in an office", "08": "summer dress in a field"}
        trained = {}

        def train(work, images, steps, out, cb):
            trained["photos"] = sorted(p.stem for p in Path(images).glob("*.png"))
            return False, "koniec testu"

        def vision(path, ch, quality=True):
            nude = Path(path).stem == "08"  # opis bez nagosci, a model wizyjny ja widzi
            return qc.Verdict(ok=not nude, reasons=[f"{qc.NUDITY} (klatka: nothing)"] if nude
                              else [])

        with tempfile.TemporaryDirectory() as tmp:
            photos = Path(tmp) / "mia" / "photos"
            photos.mkdir(parents=True)
            for name in captions:
                (photos / f"{name}.png").write_bytes(b"png")
            ch = {"slug": "mia", "name": "Mia", "trigger": "mia_aist", "steps": 10}
            with patch.object(characters, "DIR", Path(tmp)), \
                    patch.object(characters, "get", return_value=ch), \
                    patch.object(characters, "_save"), \
                    patch.object(characters.llm, "check_photo", side_effect=lambda p: {
                        "person": True, "age": 25, "gender": "woman",
                        "caption": captions[Path(p).stem]}), \
                    patch.object(characters.llm, "describe_person", return_value="blonde"), \
                    patch.object(qc, "check", side_effect=vision), \
                    patch("app.comfy_client.ComfyClient"), \
                    patch.object(characters.lora_train, "train", side_effect=train):
                characters.run_job({"slug": "mia"}, lambda f, d: None)
        self.assertEqual(trained["photos"], ["01", "03", "04", "05", "06", "07"])


class HistoryContextTests(unittest.TestCase):
    def test_the_users_order_reaches_the_history(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(history, "_HISTORY_FILE", Path(tmp) / "history.jsonl"), \
                patch.object(history.config, "HISTORY_DIR", Path(tmp)):
            history.job_context({"prompt": "x", "idea": "w bikini na plaży", "level": "nsfw"})
            history.record({"mode": "txt2img", "files": [str(Path(tmp) / "img_1.png")]})
            history.job_context(None)
            history.record({"mode": "txt2img", "files": [str(Path(tmp) / "img_2.png")]})
            first = history.entry_for("img_1.png")
            second = history.entry_for("img_2.png")
        self.assertEqual(first["idea"], "w bikini na plaży")
        self.assertNotIn("level", first)  # dawny poziom tresci nie trafia juz do historii
        self.assertNotIn("idea", second)


if __name__ == "__main__":
    unittest.main()
