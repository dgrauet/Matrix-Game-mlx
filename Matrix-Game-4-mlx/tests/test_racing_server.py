import unittest

from matrix_game_mlx.server.racing import controls, game_options


class RacingServerInputTests(unittest.TestCase):
    def test_inputs(self):
        self.assertEqual(game_options({"mode": "pursuit"})["route"], "avenue")
        self.assertEqual(controls(dict(sequence=1, drive=[1, -.5, 0, 0]))["drive"], [1, -.5, 0, 0])
        for value in ({"mode": "unknown"}, {"front_drive": 1.01}, {"front_grip": float("nan")}, {"path": "x"}):
            with self.assertRaises(ValueError):
                game_options(value)
        for value in (dict(sequence=1, drive=[1, 0, -1, 0]), dict(sequence=True), dict(sequence=1, action="exec")):
            with self.assertRaises(ValueError):
                controls(value)


if __name__ == "__main__":
    unittest.main()
