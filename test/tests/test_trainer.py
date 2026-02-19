import polars as pl
from sklearn.linear_model import LinearRegression

from microbiome_ml.train.results import CV_Result, HoldoutEvaluation
from microbiome_ml.train.trainer import ModelTrainer


class DummyFeatureSet:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def collect(self) -> pl.DataFrame:
        return self._df


class DummyDataset:
    def __init__(
        self,
        feature_df: pl.DataFrame,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> None:
        self.feature_sets = {"features": DummyFeatureSet(feature_df)}
        self._train = train_df
        self._test = test_df

    def get_train_samples(
        self, label: str, fold: None = None, metadata: bool = True
    ) -> pl.DataFrame:
        return self._train

    def get_test_samples(
        self, label: str, fold: None = None, metadata: bool = True
    ) -> pl.DataFrame:
        return self._test


def _build_simple_dataset() -> DummyDataset:
    feature_df = pl.DataFrame(
        {
            "sample": ["s1", "s2", "s3", "s4"],
            "f1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    train_df = pl.DataFrame(
        {
            "sample": ["s1", "s2"],
            "target": [2.0, 4.0],
        }
    )
    test_df = pl.DataFrame(
        {
            "sample": ["s3", "s4"],
            "target": [6.0, 8.0],
        }
    )
    return DummyDataset(feature_df, train_df, test_df)


def _build_result() -> CV_Result:
    estimator = LinearRegression()
    result = CV_Result(
        feature_set="features",
        label="target",
        scheme="holdout",
        trained_model=estimator,
    )
    return result


def test_train_and_evaluate_returns_holdout_evaluation(tmp_path):
    dataset = _build_simple_dataset()
    result = _build_result()
    trainer = ModelTrainer(
        dataset=dataset,
        best_result=result,
        output_model_path=tmp_path / "model.pkl",
    )

    evaluation = trainer.train_and_evaluate()

    assert isinstance(evaluation, HoldoutEvaluation)
    assert evaluation.predictions.shape == evaluation.targets.shape
    assert evaluation.metrics["feature_set"] == "features"
    assert evaluation.metrics["label"] == "target"
    assert evaluation.metrics["n_test"] == 2
    assert evaluation.metrics["r2"] is not None
    assert (tmp_path / "model.pkl").exists()
