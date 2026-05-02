"""Label mapping utilities for HF datasets."""

from __future__ import annotations


def infer_label_mapping(dataset_split, label_column: str) -> dict:
    """Infer label mapping from a dataset split.

    Returns:
        dict with 'id_to_label' and 'label_to_id' mappings.
    """
    feature = dataset_split.features.get(label_column)
    if feature is not None and hasattr(feature, "names"):
        names = feature.names
        id_to_label = {i: name for i, name in enumerate(names)}
        label_to_id = {name: i for i, name in enumerate(names)}
        return {"id_to_label": id_to_label, "label_to_id": label_to_id}

    # Fallback: scan unique values
    unique_labels = sorted(set(dataset_split[label_column]))
    id_to_label = {i: str(lbl) for i, lbl in enumerate(unique_labels)}
    label_to_id = {str(lbl): i for i, lbl in enumerate(unique_labels)}
    return {"id_to_label": id_to_label, "label_to_id": label_to_id}


def infer_num_classes(dataset_split, label_column: str) -> int:
    """Infer number of classes from a dataset split."""
    mapping = infer_label_mapping(dataset_split, label_column)
    return len(mapping["id_to_label"])
