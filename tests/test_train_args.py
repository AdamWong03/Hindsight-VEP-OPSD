import dataclasses
import unittest

from vep_opsd.config import HindsightOPSDConfig
from vep_opsd.train import EXECUTION_ONLY_OPSD_ARGUMENTS, OPSDArguments, DataArguments


def field_names(cls):
    return {field.name for field in dataclasses.fields(cls)}


class LaunchArgumentTest(unittest.TestCase):
    """The launcher hands OPSDArguments to the frozen recipe; the two must agree.

    Regression guard: adding a launcher-only switch (auto_resume) to
    OPSDArguments without excluding it made every run die with
    `HindsightOPSDConfig.__init__() got an unexpected keyword argument`.
    """

    def test_algorithm_fields_match_the_recipe_exactly(self):
        launch_only = set(EXECUTION_ONLY_OPSD_ARGUMENTS)
        self.assertEqual(
            field_names(OPSDArguments) - launch_only,
            field_names(HindsightOPSDConfig),
            msg=(
                "OPSDArguments and HindsightOPSDConfig diverged; add the new "
                "launcher-only switches to EXECUTION_ONLY_OPSD_ARGUMENTS"
            ),
        )

    def test_launch_only_fields_exist(self):
        for name in EXECUTION_ONLY_OPSD_ARGUMENTS:
            self.assertIn(name, field_names(OPSDArguments))
            self.assertNotIn(name, field_names(HindsightOPSDConfig))

    def test_recipe_can_be_built_from_the_cli_arguments(self):
        values = dataclasses.asdict(OPSDArguments())
        for name in EXECUTION_ONLY_OPSD_ARGUMENTS:
            values.pop(name)
        HindsightOPSDConfig(**values).validate()

    def test_data_arguments_stay_separate(self):
        self.assertEqual(
            field_names(DataArguments), {"manifest", "video_root", "image_min_pixels", "image_max_pixels"}
        )


if __name__ == "__main__":
    unittest.main()
