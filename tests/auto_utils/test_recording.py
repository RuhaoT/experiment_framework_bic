import dataclasses
import os
import auto_utils.combiparam as combiparam
import auto_utils.recording as recording


def test_save_dataclasses_to_toml():
    """1. Create a simple nested dataclass with a Combiparam field.
    2. Dump to TOML file.
    3. Load as string and check if it matches the expected output.
    """

    @dataclasses.dataclass
    class SubDataclass:
        c: list | int
        d: combiparam.Combiparam | int

    @dataclasses.dataclass
    class TestDataclass:
        a: combiparam.Combiparam | int
        b: SubDataclass

    test_dataclass = TestDataclass(
        combiparam.Combiparam([1, 2]),
        SubDataclass([3, 4], combiparam.Combiparam([5, 6])),
    )

    # Save to TOML file
    filepath = "logs/test_recording/test_toml_output.toml"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    recording.save_dataclass_to_toml(test_dataclass, filepath)

    # Load the file and check the content
    with open(filepath, "r") as f:
        content = f.read()

    expected_content = """a = [1, 2] # Type: Combiparam, Element type: int

[b]
c = [3, 4] # Type: list
d = [5, 6] # Type: Combiparam, Element type: int
"""
    assert (
        content == expected_content
    ), f"Expected:\n{expected_content}\nGot:\n{content}"

def test_load_dataclass_from_toml():
    """1. Create a simple nested dataclass with a Combiparam field.
    2. Dump to TOML file.
    3. Load from TOML file using the function.
    4. Check if the loaded dataclass matches the original.
    """

    @dataclasses.dataclass
    class SubDataclass:
        c: list | int
        d: combiparam.Combiparam | int

    @dataclasses.dataclass
    class TestDataclass:
        a: combiparam.Combiparam | int
        b: SubDataclass

    test_dataclass = TestDataclass(
        combiparam.Combiparam([1, 2]),
        SubDataclass([3, 4], combiparam.Combiparam([5, 6])),
    )

    # Save to TOML file
    filepath = "logs/test_recording/test_toml_output.toml"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    recording.save_dataclass_to_toml(test_dataclass, filepath)

    # Load from TOML file
    loaded_dataclass = recording.load_dataclass_from_toml(filepath, TestDataclass)
    
    assert loaded_dataclass == test_dataclass

def test_deserialize_combiparam_dict():
    """Tests deserialization of dictionaries with Combiparam."""
    # Test with nested dictionary
    nested_dict = {
        "a": combiparam.Combiparam([1, 2]),
        "b": {
            "c": combiparam.Combiparam([3, 4]),
            "d": 5,
            "e": {"f": combiparam.Combiparam([6, 7])},
        },
        "g": [8, 9],
    }
    expected_nested = {
        "a": [1, 2],
        "b": {"c": [3, 4], "d": 5, "e": {"f": [6, 7]}},
        "g": [8, 9],
    }
    assert recording.serialize_combiparam_dict(nested_dict) == expected_nested

    # Test with a flat dictionary
    flat_dict = {"a": combiparam.Combiparam([1, 2]), "b": 3}
    expected_flat = {"a": [1, 2], "b": 3}
    assert recording.serialize_combiparam_dict(flat_dict) == expected_flat

    # Test with an empty dictionary
    assert recording.serialize_combiparam_dict({}) == {}

    # Test with no Combiparam
    no_combiparam_dict = {"a": [1, 2], "b": 3}
    assert recording.serialize_combiparam_dict(no_combiparam_dict) == no_combiparam_dict
