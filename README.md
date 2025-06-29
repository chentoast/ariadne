Ariadne is a lightweight and minimal Python library for managing computational experiments and their results. To install, run:
```
uv add ariadne-lib
```
or
```
pip install ariadne-lib
```

> [!WARNING]
> This library is developed for personal use. Anyone is welcome to use it and give feedback, but be ready for sharp edges and breaking changes.

## Basic Usage
The primary goal of Ariadne is to help keep track of what you did and what the results were. The rapid prototyping workflow that research encourages often leads to issues with reproducibility. Plots and results can become stale and lack provenance, results may often become overwritten and lost.

The solution Ariadne provides is the simplest one possible: assign each experiment to a different unique folder, and provide simple utilities for creating and managing these folders, as well as saving useful metadata (like timestamps, notes, and configs).
The primary experiment management happens in the `Theseus` class. Calling `start` or `resume` returns a id for that experiment, along with the dedicated folder that everything should be stored under.

```python
from ariadne import Theseus

experiment = Theseus(db_path="experiments.db", base_dir="results")
experiment_id, experiment_folder = experiment.start(name="resnet", run_config={"architecture": "resnet-50"}, notes="try resnet instead of mlp")

for epoch in range(len(50)):
    x = [0, 1]
    yhat = [0, 0]
    y = [1, 1]
    loss = 1
    experiment.log({"epoch": epoch, "loss": loss})

    plt.plot(x, yhat, "-")
    plt.plot(x, y, "--")
    plt.savefig(experiment_folder / f"predictions{epoch}.png")
```

Ariadne then provides utilities for querying the experiment database by name (or an exact match by id), to quickly associate an experiment with its dedicated folder on disk.
```python
experiment_id, experiment_folder = experiment.get("resnet") # searches by substring match
experiment_id, experiment_folder = experiment.peek() # last run experiment
experiment_id = 3
experiment_folder = experiment.get_by_id(experiment_id) # or exact id match

img = plt.imread(experiment_folder / "predictions49.png")
plt.imshow(img)
```

Finally, Ariadne provides a simple cli for querying and summarizing experiments. Run `ariadne --help` for more information.

## FAQ:
Q: Why the name?

A: My project directories often end up evolving into a tangled mess of results, plots, and files, not unlike a maze. In greek mythology, Ariadne's red string was the crucial key for helping Theseus escape the labyrinth of Crete.

Q: Why not weights and biases, tensorboard, or other experiment management tools?

A: You probably should use them instead if you can! But here's a brief reason why I wrote this library instead:
- Experiments are stored locally and a simple CLI is provided. No need for a web server or a browser.
- Because all Ariadne does is handle creation and management of experiment folders, it composes well with other libraries. Your logging, plot creation and saving, and checkpointing code can remain exactly the same - just prepend the experiment folder in front of the save path.
- I simply don't need most of the features these other experiment management frameworks offer.

Q: Why not git lfs?

A: Git LFS is great for versioning large files. But, when you are rapidly iterating and creating throwaway experiments that you might not need to share, constantly writing and overwriting rich data like plots makes for horribly ugly diffs, and can bloat your cloud storage.
