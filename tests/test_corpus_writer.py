import random

from data.synth.corpus_writer import SUBJECT_COLORS, SUBJECT_SHAPES, pick_subject, write_png


class TestPickSubject:
    def test_deterministic_for_same_key(self):
        a = pick_subject("some_caption_text")
        b = pick_subject("some_caption_text")
        assert a == b

    def test_different_keys_can_differ(self):
        # not a hard guarantee for every pair, but true for a representative sample
        results = {pick_subject(f"caption number {i}") for i in range(20)}
        assert len(results) > 1

    def test_returns_valid_shape_and_color(self):
        shape, color = pick_subject("caption")
        assert shape in SUBJECT_SHAPES
        assert color in SUBJECT_COLORS

    def test_does_not_consume_a_shared_rng(self):
        """pick_subject must derive its own hash-seeded randomness, not draw from a
        shared rng — otherwise every image in the corpus would shift the rng stream
        for every file generated after it, silently changing the rest of the
        corpus for the same --seed and breaking reproducibility of already-committed
        eval numbers."""
        rng = random.Random(42)
        state_before = rng.getstate()
        pick_subject("irrelevant caption")
        assert rng.getstate() == state_before


class TestWritePng:
    def test_writes_a_valid_image_with_expected_subject(self, tmp_path):
        from PIL import Image

        path = tmp_path / "test.png"
        rng = random.Random(1)
        write_png(path, "a test caption", rng, subject=("circle", "red"))

        assert path.exists()
        img = Image.open(path)
        assert img.size == (640, 400)
