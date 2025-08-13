import atexit
import datetime
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

# TODO: figure out api for naming experiments.

@dataclass(frozen=True, slots=True)
class Spool:
    id: int
    name: str
    timestamp: datetime.datetime
    end_timestamp: datetime.datetime | None
    run_config: dict
    logs: dict
    folder: Path
    notes: str
    vc_hash: str | None = None
    vc_msg: str | None = None
    completed: bool = False

    def __str__(self):
        import pprint

        def format_ts(ts_value):
            if isinstance(ts_value, datetime.datetime):
                return ts_value.strftime("%Y-%m-%d %H:%M:%S")
            elif ts_value:
                return str(ts_value)
            return "N/A"

        def format_pformatted_dict(data_dict, max_pformat_len=250, pformat_options=None):
            if pformat_options is None:
                # Controls internal indentation of pprint and line width
                pformat_options = {"indent": 2, "width": 70, "compact": True}

            if not data_dict:
                return " None"

            s = pprint.pformat(data_dict, **pformat_options)

            truncation_suffix = "... (truncated)"

            if len(s) > max_pformat_len:
                content_allowance = max_pformat_len - len(truncation_suffix)

                if content_allowance < 10:  # Not enough space for meaningful content + suffix
                    # Replace with a simple truncation marker if pformat output is too short to cut nicely
                    s = truncation_suffix
                else:
                    # Try to cut at a newline for prettier truncation
                    # We look for a newline within the allowed content space
                    cut_at = s.rfind("\n", 0, content_allowance)
                    if cut_at != -1:  # Sensible place to cut found
                        # Add the suffix on a new line, respecting existing indent logic
                        s = s[:cut_at] + "\n" + truncation_suffix
                    else:  # No newline in the allowed part, or very long first line
                        s = s[:content_allowance] + truncation_suffix

            # Add a leading newline (to separate from the label like "Run Config:")
            # and then indent all lines of the (potentially truncated) pformat string by two spaces.
            return "\n" + "\n".join(["  " + line for line in s.splitlines()])

        status = "Complete" if self.completed else "Incomplete"
        start_ts_str = format_ts(self.timestamp)
        end_ts_str = "N/A"

        if self.completed:
            end_ts_str = format_ts(self.end_timestamp if self.end_timestamp else "N/A")

        notes_display = self.notes.strip() if self.notes else "N/A"
        if len(notes_display) > 80:
            notes_display = notes_display[:77] + "..."

        run_config_str = format_pformatted_dict(self.run_config, max_pformat_len=300)
        logs_str = format_pformatted_dict(self.logs, max_pformat_len=300)

        header = f"--- Experiment: {self.name} (ID: {self.id}) ---"
        footer = "-" * len(header)

        return f"""{header}
  Status: {status}
  Folder: {self.folder}
  Started: {start_ts_str}
  Ended: {end_ts_str}
  Notes: {notes_display}
  Config:{run_config_str}
  Logs:{logs_str}
{footer}"""


class Theseus:
    class LogLevel(IntEnum):
        NONE = 0
        INFO = 1
        DEBUG = 2

    def __init__(self, db_path: str | Path, exp_dir: str | Path, loglevel: LogLevel = LogLevel.INFO):
        self.db_path = Path(db_path).resolve()
        self.root = self.db_path.parent
        # make sure that exp_dir is a relative path, for portability
        self.exp_dir = Path(os.path.relpath(exp_dir, self.root))
        self.loglevel = loglevel

        self.__interrupted = True
        self._init_db(self.db_path)

    def _setup(self, db_id: int):
        def signal_handler(signum, frame):
            self.__interrupted = True
            sys.exit(1)

        def excepthook(exc_type, exc_value, exc_traceback):
            self.__interrupted = True
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        sys.excepthook = excepthook
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal_handler)
        atexit.register(self._cleanup, db_id)

    def _init_db(self, db_path: str | Path):
        if self.loglevel >= Theseus.LogLevel.DEBUG:
            print(f"Ariadne: Initializing database at {db_path}")

        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_timestamp DATETIME,
                    run_config TEXT,
                    logs TEXT,
                    folder TEXT NOT NULL,
                    notes TEXT,
                    vc_hash TEXT,
                    vc_msg TEXT,
                    completed BOOLEAN DEFAULT 0
                )
            """)

    def resume_or_start(self, name: str, run_config: dict, notes: str = "") -> tuple[int, Path]:
        """
        Resumes an existing, uncompleted experiment by its name or starts a new one if not found.

        If an uncompleted experiment with the given name exists, it resumes it and returns its ID and run folder.
        If not, it starts a new experiment with the given name and returns its ID and run folder.
        """
        try:
            if self.loglevel >= Theseus.LogLevel.INFO:
                print(f"Ariadne: Attempting to resume experiment '{name}'")
            return self.resume(name)
        except ValueError:
            if self.loglevel >= Theseus.LogLevel.INFO:
                print(f"Ariadne: No active experiment found with name '{name}'. Starting a new one.")
            return self.start(name, run_config, notes)

    def start(self, name: str, run_config: dict, notes: str = "") -> tuple[int, Path]:
        """
        Starts a new experiment with the given name, notes, and run configuration.
        Creates a new run folder with a timestamp and unique identifier, initializes the expbase entry,
        and registers a cleanup function to mark the experiment as completed when the program exits.
        Automatically dumps the run configuration to a JSON file in the run folder.

        Raises:
            FileExistsError: If a run folder with the same name already exists.
        """
        now = datetime.datetime.now()

        run_folder = (
            self.exp_dir / f"{name}_{now.strftime('%Y-%m-%d')}_{uuid.uuid4().hex[:8]}"
        )
        if run_folder.exists():
            raise FileExistsError(f"Run folder {run_folder} already exists.")

        db_id = None
        # for atomicity, first create a temp directory and move it to the final location later
        temp_run_folder = self.exp_dir / f"{name}.tmp_{uuid.uuid4().hex[:8]}"
        if self.loglevel >= Theseus.LogLevel.DEBUG:
            print(f"Ariadne: Creating temporary run folder for experiment '{name}' at {temp_run_folder}")
        try:
            os.makedirs(temp_run_folder)

            with open(temp_run_folder / "config.json", "w") as f:
                json.dump(run_config, f, indent=2)

            try:
                if (
                    subprocess.run(["jj"], capture_output=True, text=True, check=False).returncode
                    == 0
                ):
                    changeset, msg = get_jj_changeset_and_msg()
            except FileNotFoundError:
                changeset, msg = None, None

            try:
                if (
                    subprocess.run(
                        ["git", "rev-parse"], capture_output=True, text=True, check=False
                    ).returncode
                    == 0
                ):
                    changeset, msg = get_git_hash_and_msg()
            except FileNotFoundError:
                changeset, msg = None, None

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                res = cursor.execute(
                    """
                    INSERT INTO experiments (name, timestamp, run_config, logs, folder, notes, vc_hash, vc_msg)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
                """,
                    (
                        name,
                        datetime.datetime.now().isoformat(),
                        json.dumps(run_config),
                        json.dumps({}),
                        str(run_folder),
                        notes,
                        changeset,
                        msg,
                    ),
                )
                db_id = res.fetchone()[0]

            os.rename(temp_run_folder, run_folder)
            if self.loglevel >= Theseus.LogLevel.INFO:
                print(f"Ariadne: Started experiment '{name}' (ID: {db_id}) in folder: {run_folder}")

            self._setup(db_id)
            return db_id, run_folder.resolve()

        except Exception as e:
            if db_id is not None and run_folder.exists():
                # DB entry was created, but run folder creation failed
                if self.loglevel >= Theseus.LogLevel.INFO:
                    print(f"Ariadne: Error starting experiment '{name}': {e}. Cleaning up DB entry.")
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("DELETE FROM experiments WHERE id = ?", (db_id,))
                except sqlite3.Error as cleanup_err:
                    print(
                        f"Ariadne: CRITICAL ERROR: Failed to create run folder AND subsequently failed to clean up "
                        f"orphaned database record (ID: {db_id}). Manual intervention may be needed. "
                        f"Cleanup Error: {cleanup_err}. Original Error: {e}"
                    )

            if temp_run_folder.exists():
                shutil.rmtree(temp_run_folder)

            raise e

    def resume(self, name: str) -> tuple[int, Path]:
        """
        Resumes an existing, uncompleted experiment by its name.

        Find the most recent uncompleted experiment with the given exact name,
        re-registers the cleanup function for it, and returns its ID and run folder.
        The original run_config, notes, and other metadata are preserved.

        Raises:
            ValueError: If no uncompleted experiment with the given name is found to resume.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, folder
                FROM experiments
                WHERE name = ? AND completed = 0
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (name,),
            )
            row = cursor.fetchone()
        cursor.close()

        if row:
            db_id, folder_str = row[0], row[1]
            run_folder = Path(folder_str)

            self._setup(db_id)

            if self.loglevel >= Theseus.LogLevel.INFO:
                print(f"Ariadne: Resuming experiment '{name}' (ID: {db_id}). Original run folder: {run_folder}")
            return db_id, run_folder.resolve()
        else:
            raise ValueError(
                f"No active (uncompleted) experiment found with name '{name}' to resume."
            )

    def start_test(self) -> tuple[int, Path]:
        """
        Starts a temporary experiment.
        Creates a new run folder in the users /tmp directory. This run does not update the database entry.

        Raises:
            FileExistsError: If a run folder with the same name already exists.
        """
        now = datetime.datetime.now()
        run_folder = (Path("/tmp") / f"ariadne_test_{now.strftime('%Y-%m-%d-%H-%M-%S')}_{uuid.uuid4().hex[:4]}").resolve()

        if self.loglevel >= Theseus.LogLevel.INFO:
            print(f"Ariadne: Starting temporary experiment in {run_folder}")

        if run_folder.exists():
            raise FileExistsError(f"Run folder {run_folder} already exists.")

        try:
            db_id = -1
            os.makedirs(run_folder)

            self._setup(db_id)
            return db_id, run_folder.resolve()

        except Exception as e:
            if run_folder.exists():
                # DB entry was created, but run folder creation failed
                print(f"Ariadne: Error starting temporary experiment: {e}. Cleaning up run folder.")
                shutil.rmtree(run_folder)
            else:
                print(f"Ariadne: Error starting temporary experiment: {e}.")
            raise e


    def log(self, id: int, logs: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE experiments
                SET logs = ?
                WHERE id = ?
                """,
                (json.dumps(logs), id),
            )

    def get(self, name: str) -> list[Spool]:
        out = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            for row in conn.execute(
                """
                SELECT * FROM experiments WHERE name LIKE ?
            """,
                (f"%{name}%",),
            ):
                out.append(convert_row(row))
        return out

    def get_by_id(self, id: int) -> Spool | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM experiments WHERE id = ?
            """,
                (id,),
            )
            row = cursor.fetchone()
        cursor.close()

        if row:
            return convert_row(row)
        raise ValueError(f"No experiment found with ID {id}.")

    def peek(self) -> Spool | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM experiments ORDER BY timestamp DESC LIMIT 1
            """)
            row = cursor.fetchone()
        cursor.close()

        if row:
            return convert_row(row)
        return None

    def list(self) -> list[Spool]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            out = []
            for row in conn.execute("""
                SELECT * FROM experiments
            """):
                out.append(convert_row(row))

        return out

    def note(self, id: int, text: str, append=True):
        """
        Adds or appends a note to an experiment by its ID.
        Raises:
            ValueError: If no experiment with the given ID is found.
        """
        if append:
            current = self.get_by_id(id)
            if current and current.notes:
                text = current.notes + "\n" + text

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE experiments
                SET notes = ?
                WHERE id = ?
                """,
                (text, id),
            )

    def delete(self, id: int):
        """
        Deletes an experiment by its ID. This will remove the entry from the database and delete the associated run folder.

        Raises:
            ValueError: If no experiment with the given ID is found.
        """
        spool = self.get_by_id(id)
        if not spool:
            raise ValueError(f"No experiment found with ID {id}.")

        # Remove the database entry
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM experiments WHERE id = ?", (id,))

        # Remove the run folder
        if spool.folder.exists():
            shutil.rmtree(spool.folder)

        print(f"Experiment {id} '{spool.name}' deleted successfully.")

    def _cleanup(self, id: int):
        if self.__interrupted:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE experiments
                SET end_timestamp = ?, completed = 1
                WHERE id = ? AND completed = 0
                """,
                (datetime.datetime.now().isoformat(), id),
            )


def convert_row(row: sqlite3.Row):
    return Spool(
        id=row["id"],
        name=row["name"],
        timestamp=row["timestamp"],
        end_timestamp=row["end_timestamp"],
        run_config=json.loads(row["run_config"]),
        logs=json.loads(row["logs"]),
        folder=Path(row["folder"]),
        notes=row["notes"],
        vc_hash=row["vc_hash"],
        vc_msg=row["vc_msg"],
        completed=row["completed"],
    )


def get_jj_changeset_and_msg():
    try:
        res = subprocess.run(
            ["jj", "log", "-r", "@", "-T", "'|'++commit_id++'|'++description++'|'"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip().split("|")[1:3]
    except Exception as e:
        print(f"Error getting jj changeset: {e}")
        return None, None


def get_git_hash_and_msg():
    try:
        res = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%H\\|%B"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip().split("|")
    except Exception as e:
        print(f"Error getting git hash: {e}")
        return None, None


def cli():
    import argparse

    parser = argparse.ArgumentParser(description="Ariadne CLI")
    parser.add_argument("--db", type=str, required=True, help="Path to the SQLite database file")
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="experiments",
        help="Path to the base directory for experiments",
    )

    subparser = parser.add_subparsers(dest="command")
    subparser.add_parser("list", help="List all experiments")

    query_parser = subparser.add_parser("query", help="Get folder of an experiment by name")
    query_parser.add_argument("name", type=str, help="Name of the experiment")

    show_parser = subparser.add_parser("show", help="Show details of an experiment")
    show_parser.add_argument("id", type=int, help="ID of the experiment")
    show_parser.add_argument(
        "--fields", type=str, help="Comma separated list of fields to show (e.g., 'name', 'folder')", default="summary"
    )

    note_parser = subparser.add_parser("note", help="Annotate an experiment by ID")
    note_parser.add_argument("id", type=int, help="ID of the experiment")
    note_parser.add_argument("note", type=str, help="Note to add")
    note_parser.add_argument(
        "--append",
        action="store_true",
        help="Append the note instead of replacing existing notes",
    )

    args = parser.parse_args()

    theseus = Theseus(db_path=Path(args.db), exp_dir=Path(args.exp_dir))
    match args.command:
        case "list":
            print("\nExperiments:")
            experiments = theseus.list()
            for exp in experiments:
                print(exp.id, ":", exp.name, "->", str(exp.folder))  # Print only the name of each experiment
            print()
        case "query":
            matches = theseus.get(args.name)
            if not matches:
                print(f"No experiments found with name '{args.name}'")
                exit(1)
            for exp in matches:
                print(f"{exp.name} -> {args.exp_dir}/{exp.folder}/")
        case "show":
            import pprint as pp

            match = theseus.get_by_id(args.id)
            if not match:
                print(f"No experiments found with ID '{args.id}'")
                exit(1)

            if args.fields == "summary":
                print(str(match))
                print("-----------------")
                exit(0)

            for field in args.fields.split(","):
                pp.pprint(f"{field}: {getattr(match, field)}")
                print("-----------------")
        case "note":
            theseus.note(args.id, args.note, append=args.append)
        case _:
            parser.print_help()
            exit(1)


if __name__ == "__main__":
    cli()
