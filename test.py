import unittest
import sqlite3
import tempfile
import os
import shutil
import json
from pathlib import Path
import datetime

from ariadne import Theseus

class TestTheseus(unittest.TestCase):

    def setUp(self):
        """Set up a temporary environment for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_ariadne.db"
        self.base_exp_dir = Path(self.test_dir) / "experiments"
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.theseus = Theseus(db_path=self.db_path, base_dir=self.base_exp_dir)

    def tearDown(self):
        """Clean up the temporary environment after each test."""
        shutil.rmtree(self.test_dir)

    def test_initialization_creates_db(self):
        self.assertTrue(self.db_path.exists(), "Database file should be created.")

        # Check if the 'experiments' table exists
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiments';")
            self.assertIsNotNone(cursor.fetchone(), "Experiments table should be created.")

    def test_start_experiment(self):
        experiment_name = "test_experiment_1"
        notes = "This is a test note."
        run_config = {"param1": 10, "param2": "value2"}

        db_id, run_folder_path = self.theseus.start(name=experiment_name, notes=notes, run_config=run_config)

        self.assertIsNotNone(db_id, "Database ID should be returned.")
        self.assertTrue(Path(run_folder_path).exists(), "Run folder should be created.")
        self.assertTrue((Path(run_folder_path) / "config.json").exists(), "config.json should be created in run folder.")
        self.assertTrue((Path(run_folder_path) / "figures").exists(), "figures folder should be created in run folder.")

        # Verify config.json content
        with open(Path(run_folder_path) / "config.json", "r") as f:
            saved_config = json.load(f)
        self.assertEqual(saved_config, run_config, "Saved run_config should match input.")

        # Verify database entry
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, notes, run_config, folder, completed, end_timestamp FROM experiments WHERE id = ?", (db_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row, "Experiment should be in the database.")
            self.assertEqual(row[0], experiment_name)
            self.assertEqual(row[1], notes)
            self.assertEqual(json.loads(row[2]), run_config)
            self.assertEqual(row[3], str(run_folder_path))
            self.assertEqual(row[4], 0, "Experiment should not be completed yet.") # completed is 0 or False
            self.assertIsNone(row[5], "End timestamp should be None initially.")


    def test_03_mark_experiment(self):
        """Test marking an experiment with logs."""
        exp_name = "test_log_experiment"
        db_id, _ = self.theseus.start(name=exp_name, notes="notes", run_config={"lr": 0.01})

        logs_data = {"epoch": 1, "loss": 0.5, "accuracy": 0.8}
        self.theseus.log(id=db_id, logs=logs_data)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # The schema in the provided code has 'logs' at index 5, but 'run_config' at 4
            # The query in Theseus.get has 'metrics' (which should be logs) at index 4
            # Assuming the table schema: id[0], name[1], timestamp[2], end_timestamp[3], run_config[4], logs[5]...
            # Let's confirm the actual schema if issues arise. For now, this matches the `mark` update.
            cursor.execute("SELECT logs FROM experiments WHERE id = ?", (db_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row, "Experiment should exist.")
            saved_logs = json.loads(row[0])
            self.assertEqual(saved_logs, logs_data, "Logs data should be updated correctly.")

    def test_04_get_experiment(self):
        """Test retrieving an experiment by name."""
        exp_name = "find_me_experiment"
        notes = "Details for find_me"
        run_config = {"batch_size": 32}
        db_id_1, folder_1 = self.theseus.start(name=exp_name + "_1", notes=notes, run_config=run_config)
        # For a partial match
        db_id_2, folder_2 = self.theseus.start(name="another_" + exp_name + "_2", notes=notes, run_config=run_config)

        # Mark experiment 1 so 'metrics' (logs) exists
        self.theseus.mark(id=db_id_1, logs={"step":100})
        self.theseus.mark(id=db_id_2, logs={"step":200})


        results = self.theseus.get(name="find_me_experiment") # Should partially match both
        self.assertEqual(len(results), 2)

        # Check for the exact match first if order is not guaranteed
        found_exp_1 = False
        found_exp_2 = False
        for res in results:
            if res["id"] == db_id_1:
                self.assertEqual(res["name"], exp_name + "_1")
                self.assertEqual(res["notes"], notes)
                self.assertEqual(res["run_config"], run_config)
                self.assertEqual(res["folder"], str(folder_1))
                self.assertEqual(res["metrics"], {"step":100})
                found_exp_1 = True
            elif res["id"] == db_id_2:
                # Check the second experiment details
                self.assertEqual(res["name"], "another_" + exp_name + "_2")
                # ... other assertions for exp2 ...
                self.assertEqual(res["metrics"], {"step":200})

                found_exp_2 = True

        self.assertTrue(found_exp_1, "First experiment was not found by get.")
        self.assertTrue(found_exp_2, "Second experiment was not found by get.")


        no_results = self.theseus.get(name="non_existent_experiment")
        self.assertEqual(len(no_results), 0)

    def test_05_peek_experiment(self):
        """Test peeking the latest experiment."""
        exp1_name = "first_exp"
        exp1_id, _ = self.theseus.start(name=exp1_name, notes="old", run_config={"val":1})
        self.theseus.mark(id=exp1_id, logs={"m1":0.1})


        # Ensure timestamp is different
        import time
        time.sleep(0.01)

        exp2_name = "latest_exp"
        exp2_config = {"val": 2}
        exp2_id, exp2_folder = self.theseus.start(name=exp2_name, notes="new", run_config=exp2_config)
        exp2_logs = {"m1":0.2}
        self.theseus.mark(id=exp2_id, logs=exp2_logs)


        latest = self.theseus.peek()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["id"], exp2_id)
        self.assertEqual(latest["name"], exp2_name)
        self.assertEqual(latest["run_config"], exp2_config)
        self.assertEqual(latest["metrics"], exp2_logs)
        self.assertEqual(latest["folder"], str(exp2_folder))
        self.assertEqual(latest["notes"], "new")

    def test_06_list_experiments(self):
        """Test listing all experiment names."""
        self.theseus.start(name="exp_a", notes="", run_config={})
        self.theseus.start(name="exp_b", notes="", run_config={})
        self.theseus.start(name="exp_c", notes="", run_config={})

        names = self.theseus.list()
        self.assertIn("exp_a", names)
        self.assertIn("exp_b", names)
        self.assertIn("exp_c", names)
        self.assertEqual(len(names), 3)

    def test_cleanup_experiment(self):
        """Test the cleanup function for an experiment."""
        exp_name = "cleanup_test_exp"
        db_id, _ = self.theseus.start(name=exp_name, notes="to be cleaned", run_config={"p":1})

        # Check before cleanup
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT completed, end_timestamp FROM experiments WHERE id = ?", (db_id,))
            row_before = cursor.fetchone()
            self.assertEqual(row_before[0], 0) # completed = False/0
            self.assertIsNone(row_before[1])   # end_timestamp = NULL

        self.theseus.cleanup(db_id)

        # Check after cleanup
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT completed, end_timestamp FROM experiments WHERE id = ?", (db_id,))
            row_after = cursor.fetchone()
            self.assertEqual(row_after[0], 1) # completed = True/1
            self.assertIsNotNone(row_after[1]) # end_timestamp should be set

            # Optionally, check if the timestamp is recent
            end_time = datetime.datetime.fromisoformat(row_after[1])
            self.assertTrue((datetime.datetime.now() - end_time).total_seconds() < 5) # Within 5 seconds

        # Test that cleaning up again does not change end_timestamp or throw error (idempotency for completed = 0)
        # It actually won't update if 'completed = 0' is in WHERE, which is good.
        # Let's grab the timestamp to ensure it doesn't change.
        first_end_timestamp = row_after[1]
        self.theseus.cleanup(db_id) # Call again

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT completed, end_timestamp FROM experiments WHERE id = ?", (db_id,))
            row_again = cursor.fetchone()
            self.assertEqual(row_again[0], 1)
            self.assertEqual(row_again[1], first_end_timestamp) # Timestamp should NOT change

    def test_get_git_info_mocked(self):
        """Test that VCS info is captured (conceptual - needs mocking for subprocess)."""
        # This test would typically mock subprocess.run for git/jj commands.
        # In unittest, this means using unittest.mock.patch.
        # For now, we'll just ensure the columns exist and a None or 'unknown' is plausible.

        experiment_name = "vcs_test_experiment"
        db_id, _ = self.theseus.start(name=experiment_name, notes="vcs test", run_config={})

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vc_hash, vc_msg FROM experiments WHERE id = ?", (db_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row, "VCS info row should exist")
            # Depending on whether git/jj is available in the test env, these could be None or actual values.
            # If neither git nor jj is found by subprocess.run, they should be None as per the code.
            # If one is found but fails, 'unknown' might be there (get_git_hash_and_msg returns "unknown" on except).
            # The code sets `changeset, msg = None, None` if neither command group runs successfully.
            # And `get_jj_changeset_and_msg` returns "unknown" on error, but the main `start` logic
            # doesn't seem to propagate this "unknown" string directly if the initial `subprocess.run` for `jj` or `git rev-parse` fails.
            # Let's re-check the logic:
            # if jj works -> changeset, msg from jj
            # elif git works -> changeset, msg from git
            # else -> changeset, msg = None, None -> these are inserted into DB

            # So, if no VCS is found or subprocess fails at the check level, it should be None, None.
            self.assertIsNone(row[0], "vc_hash should be None if no VCS or error in detection")
            self.assertIsNone(row[1], "vc_msg should be None if no VCS or error in detection")

    def test_10_source_code_capture(self):
        """Test that source code of the calling frame is captured."""
        # This is tricky to test precisely without being in a specific call stack.
        # We'll check that *some* source code is captured.
        def dummy_caller():
            experiment_name = "source_code_test"
            db_id, _ = self.theseus.start(name=experiment_name, notes="src test", run_config={})
            return db_id

        db_id = dummy_caller()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT source_code FROM experiments WHERE id = ?", (db_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row, "Source code row should exist.")
            self.assertIsNotNone(row[0], "Source code string should not be None.")
            self.assertTrue(len(row[0]) > 0, "Source code string should not be empty.")
            # Check if part of `dummy_caller` or this test method is in the captured source
            self.assertIn("dummy_caller", row[0])
            self.assertIn("source_code_test", row[0])


if __name__ == '__main__':
    unittest.main()
