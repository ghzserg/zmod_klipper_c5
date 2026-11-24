"""Regression tests for exclusion around C5 extruder switches."""
import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "klippy" / "extras" / "exclude_object.py"
SPEC = importlib.util.spec_from_file_location("c5_exclude_object", str(MODULE_PATH))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeToolhead:
    name = "extruder"

    def get_extruder(self):
        return self

    def get_name(self):
        return self.name


class FakeTransform:
    def __init__(self):
        self.position = [0., 0., 0., 0.]
        self.moves = []

    def get_position(self):
        return list(self.position)

    def move(self, position, speed):
        self.position = list(position)
        self.moves.append((list(position), speed))


class FakeReactor:
    def __init__(self):
        self.pauses = []

    def monotonic(self):
        return 0.

    def pause(self, waketime):
        self.pauses.append(waketime)


class FakePrinter:
    def __init__(self):
        self.reactor = FakeReactor()

    def get_reactor(self):
        return self.reactor


class ExcludeMultiToolTest(unittest.TestCase):
    def setUp(self):
        self.transform = FakeTransform()
        self.toolhead = FakeToolhead()
        self.printer = FakePrinter()
        self.obj = MODULE.ExcludeObject.__new__(MODULE.ExcludeObject)
        self.obj.toolhead = self.toolhead
        self.obj.next_transform = self.transform
        self.obj.printer = self.printer
        self.obj._reset_state()
        self.obj.excluded_objects = ["BAD"]
        self.obj.current_object = "GOOD"
        self.obj.excluded = True
        self.obj.time_count = 0
        self.obj.xyz_offset = [0., 0., 0.]
        self.obj.physical_xyz = None
        self.obj.last_excluded_xyz = [0., 0., 0.]
        self.obj.e_states = {}
        self.obj.active_extruder = None
        self.obj.initial_extrusion_moves = 0
        self.obj.last_position = [0., 0., 0., 0.]
        self.obj.get_position()

    def use_tool(self, name, physical_e):
        self.toolhead.name = name
        self.transform.position[3] = physical_e
        return self.obj.get_position()

    def test_excluded_first_tool_does_not_extrude_on_second_tool(self):
        self.obj.move([5., 0., 0., 100.], 50.)
        self.obj.current_object = "BAD"
        self.obj.move([10., 0., 0., 140.], 50.)
        self.assertEqual(self.transform.position[3], 100.)
        virtual = self.use_tool("extruder1", 2000.)
        self.assertEqual(virtual[3], 2000.)
        self.obj.current_object = "GOOD"
        self.obj.move([20., 0., 0., 2001.], 50.)
        self.assertEqual(self.transform.position[3], 2001.)
        virtual = self.use_tool("extruder", 100.)
        self.assertEqual(virtual[3], 140.)
        self.obj.move([30., 0., 0., 141.], 50.)
        self.assertEqual(self.transform.position[3], 101.)

    def test_switching_inside_excluded_object_keeps_e_per_tool(self):
        self.obj.current_object = "BAD"
        self.obj.move([10., 0., 0., 30.], 50.)
        self.assertEqual(self.transform.position[3], 0.)
        self.assertEqual(self.use_tool("extruder1", -35000.)[3], -35000.)
        self.obj.move([20., 0., 0., -34990.], 50.)
        self.assertEqual(self.transform.position[3], -35000.)
        self.obj.current_object = "GOOD"
        self.obj.move([25., 0., 0., -34989.], 50.)
        self.assertEqual(self.transform.position[3], -34999.)
        self.assertEqual(self.use_tool("extruder", 0.)[3], 30.)
        self.obj.move([30., 0., 0., 31.], 50.)
        self.assertEqual(self.transform.position[3], 1.)

    def test_single_tool_keeps_xyz_and_skips_extrusion(self):
        self.obj.current_object = "BAD"
        self.obj.move([10., 5., 2., 30.], 50.)
        self.assertEqual(self.transform.position, [0., 0., 0., 0.])
        self.obj.current_object = "GOOD"
        self.obj.move([15., 6., 2., 31.], 50.)
        self.assertEqual(self.transform.position, [15., 6., 2., 1.])

    def test_unexcluded_tool_change_has_no_large_e_adjustment(self):
        self.obj.move([5., 0., 0., 1000.], 50.)
        self.assertEqual(self.use_tool("extruder1", 10.)[3], 10.)
        self.obj.move([6., 0., 0., 11.], 50.)
        self.assertEqual(self.transform.position[3], 11.)
        self.assertEqual(self.use_tool("extruder", 1000.)[3], 1000.)
        self.obj.move([7., 0., 0., 1001.], 50.)
        self.assertEqual(self.transform.position[3], 1001.)

    def test_c5_rate_limit_is_preserved(self):
        for step in range(1, 11):
            self.obj.move([float(step), 0., 0., float(step)], 50.)
        self.assertEqual(len(self.printer.reactor.pauses), 1)


if __name__ == "__main__":
    unittest.main()
