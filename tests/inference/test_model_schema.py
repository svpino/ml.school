import pydantic
import pytest

from inference.model import Input, Output


class TestInputSchema:
    """Tests for the Input prediction schema."""

    def test_input_defaults_are_all_none(self):
        """Every field should default to None so the schema stays fully optional."""
        assert Input().model_dump() == {
            "island": None,
            "culmen_length_mm": None,
            "culmen_depth_mm": None,
            "flipper_length_mm": None,
            "body_mass_g": None,
            "sex": None,
        }

    def test_input_accepts_full_payload(self):
        """A fully populated payload should expose exactly the expected fields."""
        sample = Input(
            island="Torgersen",
            culmen_length_mm=39.1,
            culmen_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            sex="MALE",
        )

        assert sample.model_dump().keys() == {
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        }

    def test_input_coerces_numeric_strings_to_float(self):
        """Numeric strings should be coerced to float to keep DataFrame dtypes sane."""
        sample = Input(culmen_length_mm="39.1")

        assert sample.culmen_length_mm == pytest.approx(39.1)
        assert isinstance(sample.culmen_length_mm, float)

    def test_input_rejects_non_numeric_for_float_field(self):
        """A non-numeric value for a float field should raise a validation error."""
        with pytest.raises(pydantic.ValidationError):
            Input(body_mass_g="heavy")


class TestOutputSchema:
    """Tests for the Output prediction schema."""

    def test_output_defaults_are_none(self):
        """Both fields should default to None."""
        output = Output()

        assert output.prediction is None
        assert output.confidence is None

    def test_output_accepts_values(self):
        """A populated output should round-trip its values."""
        output = Output(prediction="Adelie", confidence=0.6)

        assert output.prediction == "Adelie"
        assert output.confidence == pytest.approx(0.6)
