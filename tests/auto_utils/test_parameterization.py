import dataclasses
import auto_utils.parameterization as parameterization
import auto_utils.combiparam as combiparam

def test_iterate_dict():
    test_dict = {
        "a": combiparam.CombiParam([1, 2]),
        "b": combiparam.CombiParam([3, 4]),
        "c": list([5,6]), # this should not be combined
    }
    assert parameterization.iterate_dict(test_dict) == [
        {"a": 1, "b": 3, "c": [5,6]},
        {"a": 1, "b": 4, "c": [5,6]},
        {"a": 2, "b": 3, "c": [5,6]},
        {"a": 2, "b": 4, "c": [5,6]},
    ]

def test_recursive_iterate_dict():
    test_dict_with_subdict = {
        "a": combiparam.CombiParam([1, 2]),
        "b": {"c": combiparam.CombiParam([3, 4]), "d": combiparam.CombiParam([5, 6])},
        "e": [7, 8]
    }
    assert parameterization.recursive_iterate_dict(test_dict_with_subdict) == [
        {"a": 1, "b": {"c": 3, "d": 5}, "e": [7, 8]},
        {"a": 1, "b": {"c": 3, "d": 6}, "e": [7, 8]},
        {"a": 1, "b": {"c": 4, "d": 5}, "e": [7, 8]},
        {"a": 1, "b": {"c": 4, "d": 6}, "e": [7, 8]},
        {"a": 2, "b": {"c": 3, "d": 5}, "e": [7, 8]},
        {"a": 2, "b": {"c": 3, "d": 6}, "e": [7, 8]},
        {"a": 2, "b": {"c": 4, "d": 5}, "e": [7, 8]},
        {"a": 2, "b": {"c": 4, "d": 6}, "e": [7, 8]},
    ]

def test_recursive_iterate_dataclass():
    
    @dataclasses.dataclass
    class SubDataclass:
        c: list | int
        d: combiparam.CombiParam | int
    
    @dataclasses.dataclass
    class TestDataclass:
        a: combiparam.CombiParam | int
        b: SubDataclass
    
    test_dataclass = TestDataclass(combiparam.CombiParam([1, 2]), SubDataclass([3, 4], combiparam.CombiParam([5, 6])))
    
    assert parameterization.recursive_iterate_dataclass(test_dataclass) == [
        TestDataclass(1, SubDataclass([3,4], 5)),
        TestDataclass(1, SubDataclass([3,4], 6)),
        TestDataclass(2, SubDataclass([3,4], 5)),
        TestDataclass(2, SubDataclass([3,4], 6)),
    ]

def test_set_experiment_index():
    
    @dataclasses.dataclass
    class TestDataclass:
        a: int
        b: int
    
    param_list = [TestDataclass(1, 2), TestDataclass(3, 4)]
    param_list = parameterization.set_experiment_index(param_list)
    assert param_list[0].experiment_index == 0
    assert param_list[1].experiment_index == 1
    
    param_list = parameterization.set_experiment_index(param_list, 10)
    assert param_list[0].experiment_index == 10
    assert param_list[1].experiment_index == 11
    
    @dataclasses.dataclass
    class TestDataclassWithWrongIndex:
        a: int
        b: int
        experiment_index: str
    
    param_list = [TestDataclassWithWrongIndex(1, 2, "wrong"), TestDataclassWithWrongIndex(3, 4, "wrong")]
    try:
        parameterization.set_experiment_index(param_list)
    except ValueError as e:
        assert str(e) == "experiment index should be an integer, but got wrong"