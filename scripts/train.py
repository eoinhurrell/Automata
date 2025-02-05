import pickle

import pandas as pd
import torch

from automata.model import WorkflowTrainer


def main():
    workflows = pd.read_csv("./data/workflows_10000.csv")
    orig_steps = pd.read_csv("./data/workflow_steps_10000.csv")

    steps = orig_steps.merge(pd.read_csv("./data/valid_apps.csv"))
    steps = steps.merge(pd.read_csv("./data/valid_actions_with_triggers.csv"))
    steps["step"] = steps["app_name"] + "_" + steps["action_name"]
    step_keys = steps["step"].unique()
    pickle.dump(step_keys, open("./model/step_names.pickle", "wb"))
    num_labels = len(steps["step"].unique())

    workflows = workflows.merge(steps)
    workflows["there"] = workflows.apply(
        lambda x: x["workflow_description"]
        .lower()
        .replace(" ", "_")
        .find(x["app_name"])
        != -1,
        axis=1,
    )
    workflows = workflows[workflows["there"] == True]
    workflows = (
        workflows.groupby(["workflow_id", "workflow_description"])["step"]
        .apply(list)
        .reset_index()
    )
    workflows["len"] = workflows["step"].apply(len)
    max_steps = int(workflows["len"].max())
    workflows["label"] = workflows["step"].apply(
        lambda x: [list(step_keys).index(y) for y in x]
    )

    texts = workflows["workflow_description"].tolist()
    labels = workflows["label"].tolist()

    # Initialize trainer
    trainer = WorkflowTrainer(
        num_labels=num_labels,
        max_steps=max_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Load and split data
    train_dataset, val_dataset, test_dataset = trainer.load_data(texts, labels)

    # Train model
    trainer.train(
        train_dataset, val_dataset, batch_size=32, epochs=10, learning_rate=2e-5
    )

    # Example prediction
    sample_text = "When I receive a Gmail message, create a Trello card"
    predictions = trainer.predict(sample_text)
    print(f"Predicted workflow steps: {predictions}")


if __name__ == "__main__":
    main()
