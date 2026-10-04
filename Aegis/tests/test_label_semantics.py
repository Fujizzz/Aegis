"""Label semantics tests. No API calls, pretrained weights or CUDA required.

The dataset adapter tests use import doubles for torch/transformers only when
those packages are absent. Pandas CSV reading and production label logic are real;
these tests do not claim to validate tensor computation or neural inference.
"""
import csv
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

AEGIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AEGIS))
from utils.label_schema import get_label_schema, schema_from_config, write_schema_metadata, require_schema_metadata
from training.metrics import binary_metrics


class LabelSemanticsTests(unittest.TestCase):
    def test_pheme_keeps_rumour_as_positive_not_fake(self):
        schema = get_label_schema("pheme_rumour_v1")
        self.assertEqual([schema.encode_source(x) for x in [0, 1, "rumour", "non-rumour"]], [0, 1, 1, 0])
        prediction = schema.prediction(.9)
        self.assertEqual(prediction["final_label"], "rumour")
        self.assertEqual(prediction["p_rumour"], .9)
        self.assertNotIn("p_fake", prediction)

    def test_archive_remaps_source_real_one_to_model_fake_zero(self):
        schema = get_label_schema("archive_fake_v1")
        self.assertEqual([schema.encode_source(x) for x in [0, 1, "fake", "real"]], [1, 0, 1, 0])
        self.assertEqual(schema.prediction(.9)["final_label"], "fake")
        self.assertEqual(schema.prediction(.1)["final_label"], "real")
        self.assertAlmostEqual(schema.prediction(.1)["final_confidence"], .9)

    def test_no_implicit_config_or_cross_task_aliases(self):
        with self.assertRaises(ValueError): schema_from_config({"data": {}})
        with self.assertRaises(ValueError): get_label_schema("unknown")
        for name, invalid in [("pheme_rumour_v1", ["fake","real","true","false"]),
                              ("archive_fake_v1", ["rumour","non-rumour"])]:
            for value in invalid:
                with self.subTest(schema=name, value=value), self.assertRaises(ValueError):
                    get_label_schema(name).encode_source(value)

    def test_invalid_labels_are_never_truncated_or_filled(self):
        schema = get_label_schema("pheme_rumour_v1")
        for value in [None, "", float("nan"), float("inf"), .7, 2, -1, True, False]:
            with self.subTest(value=value), self.assertRaises(ValueError): schema.encode_source(value)
        for value in [0,1,0.0,1.0,"0.0","1.0"]:
            self.assertEqual(schema.encode_source(value), int(float(value)))

    def test_probability_boundaries_and_finite_values(self):
        schema = get_label_schema("archive_fake_v1")
        for p,label in [(0,"real"),(.4999,"real"),(.5,"fake"),(1,"fake")]:
            self.assertEqual(schema.prediction(p)["final_label"], label)
        for p in [-.1,1.1,float("nan"),float("inf")]:
            with self.assertRaises(ValueError): schema.prediction(p)

    def test_source_targets_predictions_and_actual_metrics_agree(self):
        for name, source, probs in [("archive_fake_v1", [0,1,0,1], [.9,.1,.8,.2]),
                                    ("pheme_rumour_v1", [0,1,0,1], [.1,.9,.2,.8])]:
            schema = get_label_schema(name)
            targets = np.array([schema.encode_source(x) for x in source])
            predictions = np.array([schema.prediction(x)["predicted_target"] for x in probs])
            np.testing.assert_array_equal(targets, predictions)
            metrics = binary_metrics(targets, np.array(probs))
            for key in ["accuracy","precision","recall","f1","auc"]:
                self.assertEqual(metrics[key], 1.0)

    def test_checkpoint_metadata_cannot_silently_switch_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"label_schema.json"
            pheme = get_label_schema("pheme_rumour_v1")
            archive = get_label_schema("archive_fake_v1")
            with self.assertRaises(ValueError): require_schema_metadata(path, pheme)
            write_schema_metadata(path, pheme)
            require_schema_metadata(path, pheme)
            with self.assertRaises(ValueError): require_schema_metadata(path, archive)
            with self.assertRaises(ValueError): write_schema_metadata(path, archive)
            content = json.loads(path.read_text())
            content["model_label_map"] = {"0":"rumour","1":"non_rumour"}
            path.write_text(json.dumps(content))
            with self.assertRaises(ValueError): require_schema_metadata(path, pheme)

    def test_production_dataset_applies_schema_in_every_split_without_changing_csv(self):
        doubles = {}
        if importlib.util.find_spec("torch") is None:
            torch = types.ModuleType("torch"); utils = types.ModuleType("torch.utils")
            data = types.ModuleType("torch.utils.data"); data.Dataset = object
            doubles.update({"torch":torch,"torch.utils":utils,"torch.utils.data":data})
        if importlib.util.find_spec("transformers") is None:
            transformers = types.ModuleType("transformers"); transformers.PreTrainedTokenizer = object
            doubles["transformers"] = transformers
        with patch.dict(sys.modules, doubles):
            loader = importlib.import_module("utils.data_loader")
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"source.csv"
                original = "title,text,subject,date,label,emotion_label\nfake,fixture,,,0,0.8\nreal,fixture,,,1,0.2\n"
                path.write_text(original,encoding="utf-8")
                for split in ["train","val","test"]:
                    dataset = loader.build_dataset(str(path), None, mode=split,
                        label_schema=get_label_schema("archive_fake_v1"))
                    self.assertEqual([dataset[i]["label"] for i in range(2)], [1,0])
                    self.assertEqual("emotion_label" in dataset[0], split=="train")
                self.assertEqual(path.read_text(encoding="utf-8"), original)
                self.assertNotIn('[SUBJECT] nan', dataset[0]['input_text'])
                self.assertNotIn('[DATE] nan', dataset[0]['input_text'])
                path.write_text('title,text,subject,date,label,pseudo_status\n,official text,,,0,regeneration_required\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'need verified dimension labels'):
                    loader.build_dataset(str(path), None, mode='train', label_schema=get_label_schema('pheme_rumour_v1'))
                for mode, use_pseudo in [('val', True), ('train', False)]:
                    dataset = loader.build_dataset(str(path), None, mode=mode, use_pseudo=use_pseudo,
                                                  label_schema=get_label_schema('pheme_rumour_v1'))
                    self.assertEqual(dataset[0]['input_text'], '[CONTENT]\nofficial text')

    def test_pheme_preparation_named_and_numeric_labels_agree(self):
        from utils.pheme_prepare import format_one_event_csv
        with tempfile.TemporaryDirectory() as folder:
            src=Path(folder)/"source.csv"; dst=Path(folder)/"formatted.csv"
            src.write_text("id,label,content\na,rumour,one\nb,non-rumour,two\nc,1,three\nd,0,four\n",encoding="utf-8")
            result=format_one_event_csv(str(src),str(dst),"fixture")
            self.assertEqual(result["label"].tolist(), [1,0,1,0])
            src.write_text("id,label,content\na,fake,one\n",encoding="utf-8")
            with self.assertRaises(ValueError): format_one_event_csv(str(src),str(dst),"fixture")


if __name__ == "__main__": unittest.main()
